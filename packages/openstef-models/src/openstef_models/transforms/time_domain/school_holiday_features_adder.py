# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Transform that adds school holiday features to time series data.

School holidays are retrieved from the free OpenHolidays API (https://openholidaysapi.org).
The data is licensed under the Open Database License (ODbL) 1.0, which permits commercial use
and requires attribution: "School holiday data from OpenHolidays API (openholidaysapi.org),
licensed under ODbL 1.0."

The API is only contacted in `SchoolHolidayFeatureAdder.fit` (one request per fit, unless `csv_path` is set)
and in `SchoolHolidayFeatureAdder.export_csv`. Fitted holiday periods are stored on the transform, so they
are part of the trained checkpoint. An exported file can be used without any network access, either by
setting `csv_path` (read in `fit`) or by calling `SchoolHolidayFeatureAdder.from_csv`, which returns a
fitted transform.
"""

import logging
import re
import unicodedata
from datetime import date, timedelta
from pathlib import Path
from typing import Self, override

import pandas as pd
import requests
from pydantic import ConfigDict, Field, PrivateAttr, TypeAdapter, ValidationError, field_validator, model_validator
from pydantic.alias_generators import to_camel
from pydantic_extra_types.country import CountryAlpha2
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from openstef_core.base_model import BaseConfig
from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import ConfigurationError, NotFittedError
from openstef_core.transforms import TimeSeriesTransform

_logger = logging.getLogger(__name__)


class SchoolHolidayFetchError(RuntimeError):
    """Raised when school holidays cannot be retrieved from the OpenHolidays API."""


class SchoolHolidayPeriod(BaseConfig):
    """A school holiday period. Both bounds are inclusive."""

    start_date: date = Field(description="First day of the holiday.")
    end_date: date = Field(description="Last day of the holiday.")
    name: str = Field(description="Holiday name, for example 'Summer Holidays'.")


class SchoolHolidaysRequest(BaseConfig):
    """Query parameters for the OpenHolidays `/SchoolHolidays` endpoint.

    Fields use snake_case in Python and are sent under the camelCase names the API expects.
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    country_iso_code: str = Field(description="ISO 3166-1 alpha-2 country code, for example 'NL'.")
    valid_from: date = Field(description="First date of the range (inclusive).")
    valid_to: date = Field(description="Last date of the range (inclusive).")
    subdivision_code: str | None = Field(
        default=None,
        description="ISO 3166-2 subdivision code for regional holidays, for example 'NL-NH'. None for all regions.",
    )
    language_iso_code: str = Field(default="EN", description="Language of the holiday names.")


class LocalizedText(BaseConfig):
    """A holiday name in one language."""

    language: str = Field(description="ISO 639-1 language code, for example 'EN'.")
    text: str = Field(description="The name in that language.")


class Subdivision(BaseConfig):
    """A subdivision a holiday applies to."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    code: str = Field(description="ISO 3166-2 subdivision code, for example 'NL-NH'.")
    short_name: str = Field(description="Short name of the subdivision, for example 'Noord-Holland'.")


class SchoolHolidayRecord(BaseConfig):
    """A school holiday as returned by the OpenHolidays `/SchoolHolidays` endpoint.

    Only the fields the transform needs are kept. Other response fields are ignored.
    """

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    start_date: date = Field(description="First day of the holiday.")
    end_date: date = Field(description="Last day of the holiday (inclusive).")
    name: list[LocalizedText] = Field(default_factory=list, description="The holiday name in each language.")
    nationwide: bool = Field(default=False, description="Whether the holiday applies to the whole country.")
    subdivisions: list[Subdivision] = Field(
        default_factory=list, description="Subdivisions the holiday applies to, if not nationwide."
    )

    def to_period(self) -> SchoolHolidayPeriod:
        """Convert to a period, using the English name when available."""
        english = next((text.text for text in self.name if text.language == "EN"), None)
        fallback = self.name[0].text if self.name else "School holiday"
        return SchoolHolidayPeriod(start_date=self.start_date, end_date=self.end_date, name=english or fallback)


class OpenHolidaysClient(BaseConfig):
    """Client for the OpenHolidays API school holidays endpoint.

    Holds the endpoint and transport settings. Pass a different instance (or a subclass that
    overrides `get_school_holidays`) to `SchoolHolidayFeatureAdder.client` to change how holidays
    are retrieved.

    Attribution: school holiday data from OpenHolidays API (openholidaysapi.org), licensed under ODbL 1.0.
    """

    school_holidays_url: str = Field(
        default="https://openholidaysapi.org/SchoolHolidays",
        description="URL of the OpenHolidays school holidays endpoint.",
    )
    user_agent: str = Field(
        default="openstef-school-holiday-features (+https://github.com/OpenSTEF/openstef)",
        description="User-Agent header sent with each request.",
    )
    timeout_seconds: float = Field(default=30, gt=0, description="Timeout for each HTTP request.")
    max_attempts: int = Field(default=2, ge=1, description="Maximum number of HTTP attempts per request.")
    retry_delay_seconds: float = Field(
        default=2,
        ge=0,
        description="Backoff factor between retries. The first retry is immediate; later waits grow exponentially.",
    )

    def get_school_holidays(self, request: SchoolHolidaysRequest) -> list[SchoolHolidayPeriod]:
        """Retrieve school holiday periods that overlap the requested date range.

        Makes one HTTP request. Connection errors and transient server responses (429, 5xx) are retried.

        Args:
            request: Query parameters for the endpoint.

        Returns:
            School holiday periods that overlap the requested range.

        Raises:
            SchoolHolidayFetchError: If the API cannot be reached or returns an unexpected response.
        """
        body = self._get_body(request)
        try:
            records = TypeAdapter(list[SchoolHolidayRecord]).validate_json(body)
        except ValidationError as error:
            msg = f"Unexpected response from the OpenHolidays API: {error}"
            raise SchoolHolidayFetchError(msg) from error
        return [record.to_period() for record in records]

    def _get_body(self, request: SchoolHolidaysRequest) -> bytes:
        retry = Retry(
            total=self.max_attempts - 1,
            backoff_factor=self.retry_delay_seconds,
            status_forcelist=(429, 500, 502, 503, 504),
            respect_retry_after_header=True,
        )
        with requests.Session() as session:
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            try:
                response = session.get(
                    self.school_holidays_url,
                    params=request.model_dump(by_alias=True, exclude_none=True, mode="json"),
                    headers={"Accept": "application/json", "User-Agent": self.user_agent},
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
            except requests.RequestException as error:
                msg = f"Failed to fetch school holidays from {self.school_holidays_url}: {error}"
                raise SchoolHolidayFetchError(msg) from error
        return response.content


class SchoolHolidayRow(BaseConfig):
    """One row of a CSV file written by `SchoolHolidayFeatureAdder.export_csv`.

    Field order defines the CSV columns. A row without a holiday has empty holiday fields;
    it only carries the coverage dates.
    """

    country_code: str = Field(description="Country the export was made for.")
    subdivision_code: str | None = Field(default=None, description="Subdivision the export was made for, or None.")
    covered_from: date = Field(description="First date covered by the export.")
    covered_until: date = Field(description="Last date covered by the export.")
    start_date: date | None = Field(default=None, description="First day of the holiday, or None.")
    end_date: date | None = Field(default=None, description="Last day of the holiday, or None.")
    name: str | None = Field(default=None, description="Holiday name, or None.")

    @field_validator("subdivision_code", "start_date", "end_date", "name", mode="before")
    @classmethod
    def _empty_as_none(cls, value: object) -> object:
        return value or None

    @model_validator(mode="after")
    def _holiday_fields_set_together(self) -> Self:
        values = (self.start_date, self.end_date, self.name)
        if any(value is None for value in values) and any(value is not None for value in values):
            msg = "start_date, end_date and name must either all be set or all be empty."
            raise ValueError(msg)
        return self


class SchoolHolidayFeatureAdder(BaseConfig, TimeSeriesTransform):
    """Transform that adds a school holiday indicator to time series data.

    The feature added is:

        - is_school_holiday: 1 if the day falls within a school holiday for the configured
          country (and subdivision), 0 otherwise.

    With `one_hot_by_name=True`, one more column is added per holiday name, for example
    is_school_holiday_may_holidays, with 1 on days of that holiday and 0 otherwise. The names are
    taken from the holidays known at fit time and stored with the model, sorted by name. The columns
    therefore stay the same for any data passed to `transform`, even when no holiday falls in it.
    A holiday name that was not present at fit time gets no column.

    Holidays are fetched in `fit` for the training period plus `years_ahead` years, so that
    forecasts can use holiday information for future dates. Dates outside the covered range
    are logged as a warning, because their holiday status is unknown and is reported as 0.
    Set `csv_path` to a file written by `export_csv` to read the holidays from that file in `fit` instead.

    Subdivision: for countries with regional school holidays (such as the Netherlands),
    set `subdivision_code` (for example 'NL-NH'). The API returns holidays that apply to that
    subdivision. Without it, the API returns holidays for all regions combined.

    Example:
        >>> import pandas as pd
        >>> from datetime import date, timedelta
        >>> from openstef_core.datasets import TimeSeriesDataset
        >>> from openstef_models.transforms.time_domain import SchoolHolidayFeatureAdder
        >>> from openstef_models.transforms.time_domain.school_holiday_features_adder import (
        ...     OpenHolidaysClient,
        ...     SchoolHolidayPeriod,
        ... )
        >>>
        >>> # Stub client so the example runs without network access
        >>> class StubClient(OpenHolidaysClient):
        ...     def get_school_holidays(self, request):
        ...         christmas = SchoolHolidayPeriod(
        ...             start_date=date(2025, 12, 24), end_date=date(2025, 12, 26), name="Christmas"
        ...         )
        ...         return [christmas]
        >>> transform = SchoolHolidayFeatureAdder(country_code="NL", subdivision_code="NL-NH", client=StubClient())
        >>> data = pd.DataFrame({
        ...     'load': [100] * 5
        ... }, index=pd.date_range('2025-12-24', periods=5, freq='D'))
        >>> dataset = TimeSeriesDataset(data, sample_interval=timedelta(days=1))
        >>> transform.fit(dataset)
        >>> transform.transform(dataset).data["is_school_holiday"].tolist()
        [1, 1, 1, 0, 0]
    """

    country_code: CountryAlpha2 = Field(description="Country code for school holidays, for example 'NL'.")
    subdivision_code: str | None = Field(
        default=None,
        description="ISO 3166-2 subdivision code for regional school holidays, for example 'NL-NH'.",
    )
    years_ahead: int = Field(
        default=2,
        ge=0,
        description="Number of years after the training period for which holidays are fetched in fit.",
    )
    feature_name: str = Field(default="is_school_holiday", description="Name of the feature column added.")
    one_hot_by_name: bool = Field(
        default=False,
        description="Also add one 0/1 column per holiday name, named '<feature_name>_<name slug>'.",
    )
    csv_path: Path | None = Field(
        default=None,
        description="CSV written by export_csv. If set, fit reads the holidays from this file instead of the API.",
    )
    client: OpenHolidaysClient = Field(
        default_factory=OpenHolidaysClient,
        exclude=True,
        description="Client used to fetch holidays. Not stored in the checkpoint.",
    )

    _covered_from: date | None = PrivateAttr(default=None)
    _covered_until: date | None = PrivateAttr(default=None)
    _holidays: list[SchoolHolidayPeriod] = PrivateAttr(default_factory=list)
    _is_fitted: bool = PrivateAttr(default=False)
    _holiday_slugs: list[str] = PrivateAttr(default_factory=list)

    @property
    def covered_from(self) -> date | None:
        """First date covered by the holidays, or None when a CSV without holidays was loaded."""
        return self._covered_from

    @property
    def covered_until(self) -> date | None:
        """Last date covered by the holidays, or None when a CSV without holidays was loaded."""
        return self._covered_until

    @property
    def holidays(self) -> list[SchoolHolidayPeriod]:
        """School holiday periods known to the transform."""
        return list(self._holidays)

    @property
    @override
    def is_fitted(self) -> bool:
        return self._is_fitted

    @override
    def fit(self, data: TimeSeriesDataset) -> None:
        """Fetch school holidays for the training period and `years_ahead` years after it.

        If `csv_path` is set, the holidays and their coverage dates are read from that file instead,
        and the API is not contacted. Otherwise makes exactly one request to the OpenHolidays API.

        Args:
            data: Time series dataset whose index defines the training period.

        Raises:
            SchoolHolidayFetchError: If the holidays cannot be retrieved.
            ConfigurationError: If `csv_path` was exported for a different country or subdivision.
        """
        if self.is_fitted:
            return

        if self.csv_path is not None:
            self._load_csv(path=self.csv_path)
            return

        start = data.index.min().date()
        end = (data.index.max() + pd.DateOffset(years=self.years_ahead)).date()
        holidays = self.client.get_school_holidays(self._request(start, end))
        self._set_fitted_state(holidays, covered_from=start, covered_until=end)

    @override
    def transform(self, data: TimeSeriesDataset) -> TimeSeriesDataset:
        """Add the school holiday indicator. Does not contact the API.

        Args:
            data: Input time series dataset.

        Returns:
            Transformed dataset with the school holiday feature added.

        Raises:
            NotFittedError: If the transform has neither been fitted nor loaded from CSV.
        """
        if not self.is_fitted:
            raise NotFittedError(self.__class__.__name__)

        dates = pd.Series(data.index.date, index=data.index)
        covered_from, covered_until = self.covered_from, self.covered_until
        if (
            covered_from is not None
            and covered_until is not None
            and (dates.min() < covered_from or dates.max() > covered_until)
        ):
            _logger.warning(
                "Input dates fall outside the school holiday coverage %s to %s; "
                "dates outside this range are not flagged as holidays.",
                covered_from,
                covered_until,
            )

        features = {
            column: dates.isin(self._expand_dates(periods)).astype(int).to_numpy()
            for column, periods in self._feature_columns().items()
        }
        return TimeSeriesDataset(
            data=data.data.assign(**features),
            sample_interval=data.sample_interval,
        )

    @override
    def features_added(self) -> list[str]:
        return list(self._feature_columns())

    def _feature_columns(self) -> dict[str, list[SchoolHolidayPeriod]]:
        """Map each feature column to the holiday periods that set it to 1.

        The one-hot columns come from the names frozen at fit time, so they do not depend on
        the holidays that happen to fall in the data being transformed.
        """
        columns: dict[str, list[SchoolHolidayPeriod]] = {self.feature_name: self._holidays}
        if self.one_hot_by_name:
            for slug in self._holiday_slugs:
                columns[f"{self.feature_name}_{slug}"] = [
                    period for period in self._holidays if self._name_slug(period.name) == slug
                ]
        return columns

    def export_csv(self, path: Path, start: date, end: date) -> None:
        """Fetch school holidays for a date range and write them to a CSV file.

        Makes exactly one request to the OpenHolidays API. The file can be loaded later with
        `from_csv`, which needs no network access. Export a range that covers the period you
        intend to use, because dates outside the exported range are not flagged as holidays.

        A range without any holidays writes a single row with the coverage dates and empty holiday
        fields, so the file still loads with `from_csv`.

        Args:
            path: Destination CSV file. An existing file is overwritten.
            start: First date to export.
            end: Last date to export.

        Raises:
            ValueError: If `end` is before `start`.
            SchoolHolidayFetchError: If the holidays cannot be retrieved.
        """
        if end < start:
            msg = f"end ({end}) must not be before start ({start})."
            raise ValueError(msg)

        periods = self.client.get_school_holidays(self._request(start, end))
        rows = [
            SchoolHolidayRow(
                country_code=self.country_code,
                subdivision_code=self.subdivision_code,
                covered_from=start,
                covered_until=end,
                start_date=period.start_date,
                end_date=period.end_date,
                name=period.name,
            )
            for period in periods
        ]
        if not rows:
            # Coverage dates are stored on every row, so an export without holidays still needs one row.
            rows.append(
                SchoolHolidayRow(
                    country_code=self.country_code,
                    subdivision_code=self.subdivision_code,
                    covered_from=start,
                    covered_until=end,
                )
            )
        frame = pd.DataFrame(
            [row.model_dump(mode="json") for row in rows],
            columns=list(SchoolHolidayRow.model_fields),
        )
        frame.to_csv(path, index=False)

    @classmethod
    def from_csv(
        cls,
        path: Path,
        country_code: CountryAlpha2,
        subdivision_code: str | None = None,
        *,
        one_hot_by_name: bool = False,
    ) -> Self:
        """Create a transform from a CSV file written by `export_csv`.

        Makes no network requests, so the transform does not need to be trained. A file without
        any rows (a header only, or no content at all) means no holidays are known. The returned
        transform is fitted, adds 0 for every date, and has no coverage range to check dates against.
        Such a file cannot state its country or subdivision, so those are not checked.

        Args:
            path: CSV file written by `export_csv`.
            country_code: Country the file must have been exported for.
            subdivision_code: Subdivision the file must have been exported for, or None.
            one_hot_by_name: Also add one 0/1 column per holiday name. See the class docstring.

        Returns:
            A transform covering the date range of the file, with the holidays in the file.

        Raises:
            ConfigurationError: If the file is malformed or was exported for a different
                country or subdivision.
        """
        transform = cls(
            country_code=country_code,
            subdivision_code=subdivision_code,
            one_hot_by_name=one_hot_by_name,
            csv_path=path,
        )
        transform._load_csv(path)
        return transform

    def _load_csv(self, path: Path) -> None:
        """Set the coverage dates and holidays from a CSV file written by `export_csv`.

        Raises:
            ConfigurationError: If the file is malformed or was exported for a different country or subdivision.
        """
        try:
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        except pd.errors.EmptyDataError:
            frame = pd.DataFrame(columns=list(SchoolHolidayRow.model_fields))

        try:
            rows = [SchoolHolidayRow.model_validate(record) for record in frame.to_dict(orient="records")]
        except ValidationError as error:
            msg = f"{path} is not a school holiday export: {error}"
            raise ConfigurationError(msg) from error

        if not rows:
            self._set_fitted_state([], covered_from=None, covered_until=None)
            return

        first = rows[0]
        if first.country_code != self.country_code or first.subdivision_code != self.subdivision_code:
            msg = (
                f"{path} was exported for country {first.country_code} and subdivision {first.subdivision_code}, "
                f"but country {self.country_code} and subdivision {self.subdivision_code} are configured."
            )
            raise ConfigurationError(msg)
        holidays = [
            SchoolHolidayPeriod(start_date=row.start_date, end_date=row.end_date, name=row.name)
            for row in rows
            if row.start_date is not None and row.end_date is not None and row.name is not None
        ]
        self._set_fitted_state(holidays, covered_from=first.covered_from, covered_until=first.covered_until)

    def _set_fitted_state(
        self,
        holidays: list[SchoolHolidayPeriod],
        *,
        covered_from: date | None,
        covered_until: date | None,
    ) -> None:
        """Store the holidays and freeze the one-hot column names seen at fit time."""
        self._holidays = holidays
        self._covered_from = covered_from
        self._covered_until = covered_until
        self._holiday_slugs = sorted({self._name_slug(period.name) for period in holidays})
        self._is_fitted = True

    def _request(self, start: date, end: date) -> SchoolHolidaysRequest:
        return SchoolHolidaysRequest(
            country_iso_code=self.country_code,
            valid_from=start,
            valid_to=end,
            subdivision_code=self.subdivision_code,
        )

    @staticmethod
    def _expand_dates(periods: list[SchoolHolidayPeriod]) -> set[date]:
        return {
            period.start_date + timedelta(days=offset)
            for period in periods
            for offset in range((period.end_date - period.start_date).days + 1)
        }

    @staticmethod
    def _name_slug(name: str) -> str:
        ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
        return re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_") or "unnamed"


class SchoolHolidayConfig(BaseConfig):
    """Opt-in configuration for school holiday features.

    Set on a workflow config to add school holiday features. Holidays are fetched once during fit, or read from
    csv_path, and then frozen in the model checkpoint so the feature set stays stable.
    """

    country_code: CountryAlpha2 = Field(description="Country code for school holidays, for example 'NL'.")
    subdivision_code: str | None = Field(
        default=None,
        description="ISO 3166-2 subdivision code for regional school holidays, for example 'NL-NH'.",
    )
    years_ahead: int = Field(
        default=2,
        ge=0,
        description="Number of years after the training period for which holidays are fetched in fit.",
    )
    feature_name: str = Field(default="is_school_holiday", description="Name of the feature column added.")
    one_hot_by_name: bool = Field(
        default=False,
        description="Also add one 0/1 column per holiday name, named '<feature_name>_<name slug>'.",
    )
    csv_path: Path | None = Field(
        default=None,
        description="CSV written by SchoolHolidayFeatureAdder.export_csv. If set, fit reads it instead of the API.",
    )

    def create_transform(self) -> SchoolHolidayFeatureAdder:
        """Create a new school holiday transform with this configuration.

        Returns:
            A fresh, unfitted SchoolHolidayFeatureAdder. Each workflow gets its own instance.
        """
        return SchoolHolidayFeatureAdder(
            country_code=self.country_code,
            subdivision_code=self.subdivision_code,
            years_ahead=self.years_ahead,
            feature_name=self.feature_name,
            one_hot_by_name=self.one_hot_by_name,
            csv_path=self.csv_path,
        )
