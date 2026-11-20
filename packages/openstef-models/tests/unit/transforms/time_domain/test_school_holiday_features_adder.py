# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Unit tests for SchoolHolidayFeatureAdder and OpenHolidaysClient. No test contacts the OpenHolidays API."""

import json
import logging
import pickle
import socket
import threading
import urllib.parse
from collections import deque
from collections.abc import Iterator
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, override
from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import Field, ValidationError
from pydantic_extra_types.country import CountryAlpha2

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import ConfigurationError, NotFittedError
from openstef_core.mixins.transform import TransformPipeline
from openstef_models.transforms.time_domain import SchoolHolidayFeatureAdder
from openstef_models.transforms.time_domain.school_holiday_features_adder import (
    OpenHolidaysClient,
    SchoolHolidayConfig,
    SchoolHolidayFetchError,
    SchoolHolidayPeriod,
    SchoolHolidaysRequest,
)


def _api_item(start: str, end: str, names: list[dict[str, str]] | None = None) -> dict[str, Any]:
    return {
        "id": "2b4c1f0e",
        "startDate": start,
        "endDate": end,
        "type": "SchoolHolidays",
        "name": names
        if names is not None
        else [{"language": "NL", "text": "Zomervakantie"}, {"language": "EN", "text": "Summer Holidays"}],
        "nationwide": False,
        "subdivisions": [{"code": "NL-NH", "shortName": "NH"}],
        "regionalScope": "Regional",
        "temporalScope": "FullDay",
    }


def _body(*items: dict[str, Any]) -> bytes:
    return json.dumps(list(items)).encode()


def _dataset(start: str, periods: int) -> TimeSeriesDataset:
    index = pd.date_range(start, periods=periods, freq="D")
    return TimeSeriesDataset(
        pd.DataFrame({"load": range(periods)}, index=index),
        sample_interval=timedelta(days=1),
    )


def _summer() -> SchoolHolidayPeriod:
    return SchoolHolidayPeriod(start_date=date(2025, 7, 1), end_date=date(2025, 7, 3), name="Summer Holidays")


def _may() -> SchoolHolidayPeriod:
    return SchoolHolidayPeriod(start_date=date(2025, 5, 1), end_date=date(2025, 5, 2), name="May Holidays")


def _nationwide(period: SchoolHolidayPeriod) -> SchoolHolidayPeriod:
    """Expected form of a period without subdivisions after fit or CSV load: the only group, index 0."""
    return period.model_copy(update={"group_code": "0"})


def _unreachable_url() -> str:
    """Return a localhost URL with nothing listening, so connections are refused."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}/SchoolHolidays"


class _HolidayServer:
    """Local stand-in for the OpenHolidays API.

    Replies to each request with the next queued (status, body) pair, or HTTP 500 once the queue is empty.
    Records the path, query, and User-Agent of every request.
    """

    def __init__(self) -> None:
        self.responses: deque[tuple[int, bytes]] = deque()
        self.received: list[tuple[str, dict[str, list[str]], str | None]] = []
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self.url = f"http://127.0.0.1:{self._httpd.server_port}/SchoolHolidays"

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parsed = urllib.parse.urlsplit(self.path)
                server.received.append((parsed.path, urllib.parse.parse_qs(parsed.query), self.headers["User-Agent"]))
                status, body = server.responses.popleft() if server.responses else (500, b"")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        return Handler

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture
def holiday_server() -> Iterator[_HolidayServer]:
    server = _HolidayServer()
    server.start()
    try:
        yield server
    finally:
        server.stop()


class _RecordingClient(OpenHolidaysClient):
    """Client that returns canned periods and records each request instead of calling the API."""

    periods: list[SchoolHolidayPeriod] = Field(default_factory=list)
    requests: list[SchoolHolidaysRequest] = Field(default_factory=list)

    @override
    def get_school_holidays(self, request: SchoolHolidaysRequest) -> list[SchoolHolidayPeriod]:
        self.requests.append(request)
        return list(self.periods)


def _fitted(
    holidays: list[SchoolHolidayPeriod] | None = None,
    *,
    start: str = "2025-01-01",
    end: str = "2025-12-31",
    **config: Any,
) -> SchoolHolidayFeatureAdder:
    """Return an adder fitted through fit(), covering exactly start..end (years_ahead=0)."""
    client = _RecordingClient(periods=holidays or [])
    transform = SchoolHolidayFeatureAdder(country_code="NL", years_ahead=0, client=client, **config)
    days = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
    transform.fit(_dataset(start, periods=days))
    return transform


class TestOpenHolidaysClient:
    """Tests for the HTTP client, its request model, and its response parsing."""

    def test_default_url_is_the_documented_endpoint(self) -> None:
        assert OpenHolidaysClient().school_holidays_url == "https://openholidaysapi.org/SchoolHolidays"

    def test_sends_documented_query_and_user_agent(self, holiday_server: _HolidayServer) -> None:
        # Arrange
        client = OpenHolidaysClient(school_holidays_url=holiday_server.url)
        holiday_server.responses.append((200, _body()))
        request = SchoolHolidaysRequest(
            country_iso_code="NL",
            valid_from=date(2025, 1, 1),
            valid_to=date(2025, 12, 31),
            subdivision_code="NL-NH",
        )

        # Act
        client.get_school_holidays(request)

        # Assert
        assert len(holiday_server.received) == 1
        path, query, user_agent = holiday_server.received[0]
        assert path == "/SchoolHolidays"
        assert query == {
            "countryIsoCode": ["NL"],
            "validFrom": ["2025-01-01"],
            "validTo": ["2025-12-31"],
            "subdivisionCode": ["NL-NH"],
            "languageIsoCode": ["EN"],
        }
        assert user_agent == client.user_agent

    def test_omits_subdivision_when_not_set(self, holiday_server: _HolidayServer) -> None:
        # Arrange
        client = OpenHolidaysClient(school_holidays_url=holiday_server.url)
        holiday_server.responses.append((200, _body()))
        request = SchoolHolidaysRequest(country_iso_code="DE", valid_from=date(2025, 1, 1), valid_to=date(2025, 12, 31))

        # Act
        client.get_school_holidays(request)

        # Assert
        _, query, _ = holiday_server.received[0]
        assert "subdivisionCode" not in query

    def test_parses_response_into_periods(self, holiday_server: _HolidayServer) -> None:
        # Arrange
        client = OpenHolidaysClient(school_holidays_url=holiday_server.url)
        holiday_server.responses.append((200, _body(_api_item("2025-07-05", "2025-08-31"))))
        request = SchoolHolidaysRequest(country_iso_code="NL", valid_from=date(2025, 1, 1), valid_to=date(2025, 12, 31))

        # Act
        periods = client.get_school_holidays(request)

        # Assert
        assert periods == [
            SchoolHolidayPeriod(
                start_date=date(2025, 7, 5),
                end_date=date(2025, 8, 31),
                name="Summer Holidays",
                subdivisions=["NL-NH"],
            )
        ]

    @pytest.mark.parametrize(
        ("names", "expected"),
        [
            pytest.param([{"language": "EN", "text": "Summer Holidays"}], "Summer Holidays", id="english"),
            pytest.param(
                [{"language": "NL", "text": "Zomervakantie"}, {"language": "EN", "text": "Summer Holidays"}],
                "Summer Holidays",
                id="english-among-others",
            ),
            pytest.param([{"language": "NL", "text": "Zomervakantie"}], "Zomervakantie", id="no-english-uses-first"),
            pytest.param([], "School holiday", id="no-names"),
        ],
    )
    def test_chooses_holiday_name(
        self, holiday_server: _HolidayServer, names: list[dict[str, str]], expected: str
    ) -> None:
        # Arrange
        client = OpenHolidaysClient(school_holidays_url=holiday_server.url)
        holiday_server.responses.append((200, _body(_api_item("2025-07-05", "2025-08-31", names))))
        request = SchoolHolidaysRequest(country_iso_code="NL", valid_from=date(2025, 1, 1), valid_to=date(2025, 12, 31))

        # Act
        periods = client.get_school_holidays(request)

        # Assert
        assert [period.name for period in periods] == [expected]

    @pytest.mark.parametrize("status", [429, 500, 503], ids=["rate-limited", "server-error", "unavailable"])
    def test_retries_transient_http_error(self, holiday_server: _HolidayServer, status: int) -> None:
        # Arrange
        client = OpenHolidaysClient(school_holidays_url=holiday_server.url, retry_delay_seconds=0)
        holiday_server.responses.extend([(status, b""), (200, _body(_api_item("2025-07-05", "2025-08-31")))])
        request = SchoolHolidaysRequest(country_iso_code="NL", valid_from=date(2025, 1, 1), valid_to=date(2025, 12, 31))

        # Act
        periods = client.get_school_holidays(request)

        # Assert
        assert len(periods) == 1
        assert len(holiday_server.received) == 2

    def test_connection_error_raises_fetch_error(self) -> None:
        # Arrange
        client = OpenHolidaysClient(school_holidays_url=_unreachable_url(), max_attempts=2, retry_delay_seconds=0)
        request = SchoolHolidaysRequest(country_iso_code="NL", valid_from=date(2025, 1, 1), valid_to=date(2025, 12, 31))

        # Act / Assert
        with pytest.raises(SchoolHolidayFetchError):
            client.get_school_holidays(request)

    def test_raises_after_all_attempts_fail(self, holiday_server: _HolidayServer) -> None:
        # Arrange
        client = OpenHolidaysClient(school_holidays_url=holiday_server.url, max_attempts=3, retry_delay_seconds=0)
        holiday_server.responses.extend([(503, b"")] * 3)
        request = SchoolHolidaysRequest(country_iso_code="NL", valid_from=date(2025, 1, 1), valid_to=date(2025, 12, 31))

        # Act / Assert
        with pytest.raises(SchoolHolidayFetchError):
            client.get_school_holidays(request)
        assert len(holiday_server.received) == 3

    def test_client_error_is_not_retried(self, holiday_server: _HolidayServer) -> None:
        # Arrange
        client = OpenHolidaysClient(school_holidays_url=holiday_server.url, retry_delay_seconds=0)
        holiday_server.responses.extend([(404, b""), (200, _body(_api_item("2025-07-05", "2025-08-31")))])
        request = SchoolHolidaysRequest(country_iso_code="NL", valid_from=date(2025, 1, 1), valid_to=date(2025, 12, 31))

        # Act / Assert
        with pytest.raises(SchoolHolidayFetchError):
            client.get_school_holidays(request)
        assert len(holiday_server.received) == 1

    @pytest.mark.parametrize(
        "body",
        [
            pytest.param(b"not json", id="not-json"),
            pytest.param(b'{"startDate": "2025-07-05"}', id="not-a-list"),
            pytest.param(b'[{"startDate": "2025-07-05"}]', id="missing-end-date"),
        ],
    )
    def test_invalid_response_raises_without_retry(self, holiday_server: _HolidayServer, body: bytes) -> None:
        # Arrange
        client = OpenHolidaysClient(school_holidays_url=holiday_server.url)
        holiday_server.responses.append((200, body))
        request = SchoolHolidaysRequest(country_iso_code="NL", valid_from=date(2025, 1, 1), valid_to=date(2025, 12, 31))

        # Act / Assert
        with pytest.raises(SchoolHolidayFetchError):
            client.get_school_holidays(request)
        assert len(holiday_server.received) == 1


class TestSchoolHolidayFeatureAdder:
    """Tests for fitting, transforming, and CSV import and export."""

    def test_fit_makes_one_request_for_training_period_and_years_ahead(self) -> None:
        # Arrange
        client = _RecordingClient(periods=[_summer()])
        transform = SchoolHolidayFeatureAdder(country_code="NL", subdivision_code="NL-NH", years_ahead=2, client=client)

        # Act
        transform.fit(_dataset("2025-01-01", periods=10))

        # Assert
        assert len(client.requests) == 1
        request = client.requests[0]
        assert request.valid_from == date(2025, 1, 1)
        assert request.valid_to == date(2027, 1, 10)
        assert request.subdivision_code == "NL-NH"
        assert transform.covered_from == date(2025, 1, 1)
        assert transform.covered_until == date(2027, 1, 10)
        assert transform.holidays == [_nationwide(_summer())]

    def test_fit_with_default_client_makes_one_http_request(self, holiday_server: _HolidayServer) -> None:
        # Arrange
        client = OpenHolidaysClient(school_holidays_url=holiday_server.url)
        transform = SchoolHolidayFeatureAdder(country_code="NL", client=client)
        holiday_server.responses.append((200, _body(_api_item("2025-07-05", "2025-08-31"))))

        # Act
        transform.fit(_dataset("2025-01-01", periods=10))

        # Assert
        assert len(holiday_server.received) == 1
        assert transform.is_fitted

    def test_fit_propagates_fetch_error(self) -> None:
        # Arrange
        client = OpenHolidaysClient(school_holidays_url=_unreachable_url(), retry_delay_seconds=0)
        transform = SchoolHolidayFeatureAdder(country_code="NL", client=client)

        # Act / Assert
        with pytest.raises(SchoolHolidayFetchError):
            transform.fit(_dataset("2025-01-01", periods=10))
        assert not transform.is_fitted

    def test_transform_before_fit_raises(self) -> None:
        # Arrange
        transform = SchoolHolidayFeatureAdder(country_code="NL")

        # Act / Assert
        with pytest.raises(NotFittedError):
            transform.transform(_dataset("2025-07-01", periods=3))

    def test_transform_does_not_contact_client(self) -> None:
        # Arrange
        client = _RecordingClient(periods=[_summer()])
        transform = SchoolHolidayFeatureAdder(country_code="NL", client=client)
        transform.fit(_dataset("2025-01-01", periods=10))

        # Act
        transform.transform(_dataset("2025-07-01", periods=3))

        # Assert
        assert len(client.requests) == 1

    def test_flags_both_holiday_bounds_inclusive(self) -> None:
        # Arrange
        transform = _fitted([_summer()])

        # Act
        result = transform.transform(_dataset("2025-06-30", periods=5))

        # Assert
        assert result.data["is_school_holiday"].tolist() == [0, 1, 1, 1, 0]

    def test_adds_configured_feature_name(self) -> None:
        # Arrange
        transform = _fitted(feature_name="school_holiday")

        # Act
        result = transform.transform(_dataset("2025-07-01", periods=2))

        # Assert
        assert transform.features_added() == ["school_holiday"]
        assert "school_holiday" in result.data.columns

    def test_one_hot_disabled_by_default(self) -> None:
        # Arrange
        transform = _fitted([_summer(), _may()])

        # Act
        result = transform.transform(_dataset("2025-04-30", periods=5))

        # Assert
        assert transform.features_added() == ["is_school_holiday"]
        assert list(result.data.columns) == ["load", "is_school_holiday"]

    def test_one_hot_adds_column_per_holiday_name(self) -> None:
        # Arrange
        transform = _fitted([_summer(), _may()], one_hot_by_name=True)

        # Act
        result = transform.transform(_dataset("2025-04-30", periods=5))

        # Assert
        assert transform.features_added() == [
            "is_school_holiday",
            "is_school_holiday_may_holidays",
            "is_school_holiday_summer_holidays",
        ]
        assert result.data["is_school_holiday"].tolist() == [0, 1, 1, 0, 0]
        assert result.data["is_school_holiday_may_holidays"].tolist() == [0, 1, 1, 0, 0]
        assert result.data["is_school_holiday_summer_holidays"].tolist() == [0, 0, 0, 0, 0]

    def test_one_hot_columns_do_not_depend_on_holiday_input_order(self) -> None:
        # Arrange
        forward = _fitted([_summer(), _may()], one_hot_by_name=True)
        reversed_order = _fitted([_may(), _summer()], one_hot_by_name=True)

        # Act / Assert
        assert forward.features_added() == reversed_order.features_added()

    def test_one_hot_columns_stay_for_window_without_holidays(self) -> None:
        # Arrange
        transform = _fitted([_summer(), _may()], one_hot_by_name=True)

        # Act
        result = transform.transform(_dataset("2025-01-01", periods=3))

        # Assert
        assert transform.features_added() == [
            "is_school_holiday",
            "is_school_holiday_may_holidays",
            "is_school_holiday_summer_holidays",
        ]
        assert result.data["is_school_holiday_may_holidays"].tolist() == [0, 0, 0]
        assert result.data["is_school_holiday_summer_holidays"].tolist() == [0, 0, 0]

    def test_one_hot_groups_periods_with_the_same_name(self) -> None:
        # Arrange
        summer_2026 = SchoolHolidayPeriod(
            start_date=date(2026, 7, 1), end_date=date(2026, 7, 2), name="Summer Holidays"
        )
        transform = _fitted([_summer(), summer_2026], end="2026-12-31", one_hot_by_name=True)

        # Act
        result = transform.transform(_dataset("2026-06-30", periods=4))

        # Assert
        assert transform.features_added() == ["is_school_holiday", "is_school_holiday_summer_holidays"]
        assert result.data["is_school_holiday_summer_holidays"].tolist() == [0, 1, 1, 0]

    @pytest.mark.parametrize(
        ("name", "expected_column"),
        [
            pytest.param("Zomervakantie (NL-NH)", "is_school_holiday_zomervakantie_nl_nh", id="punctuation"),
            pytest.param("Vacances d'été", "is_school_holiday_vacances_d_ete", id="accents"),
            pytest.param("!!!", "is_school_holiday_unnamed", id="no-letters"),
        ],
    )
    def test_one_hot_column_name_is_slugified(self, name: str, expected_column: str) -> None:
        # Arrange
        transform = _fitted(
            [SchoolHolidayPeriod(start_date=date(2025, 7, 1), end_date=date(2025, 7, 2), name=name)],
            one_hot_by_name=True,
        )

        # Act
        features = transform.features_added()

        # Assert
        assert features == ["is_school_holiday", expected_column]

    def test_warns_for_dates_outside_coverage(self, caplog: pytest.LogCaptureFixture) -> None:
        # Arrange
        transform = _fitted()

        # Act
        with caplog.at_level(logging.WARNING):
            transform.transform(_dataset("2026-01-01", periods=2))

        # Assert
        assert "outside the school holiday coverage" in caplog.text

    def test_pipeline_does_not_refetch_loaded_transform(self, tmp_path: Path) -> None:
        # Arrange
        path = tmp_path / "holidays.csv"
        SchoolHolidayFeatureAdder(country_code="NL", client=_RecordingClient(periods=[_summer()])).export_csv(
            path, start=date(2025, 1, 1), end=date(2025, 12, 31)
        )
        loaded = SchoolHolidayFeatureAdder.from_csv(path, country_code=CountryAlpha2("NL"))

        # Act / Assert
        with patch("requests.Session.send", side_effect=AssertionError("network access")):
            TransformPipeline(transforms=[loaded]).fit(_dataset("2025-07-01", periods=3))

    def test_client_is_not_serialised(self) -> None:
        # Arrange
        transform = SchoolHolidayFeatureAdder(country_code="NL")

        # Act / Assert
        assert "client" not in transform.model_dump()


class TestSchoolHolidayCsv:
    """Tests for exporting holidays to CSV and loading them back."""

    def test_export_makes_one_request_and_writes_rows(self, tmp_path: Path) -> None:
        # Arrange
        path = tmp_path / "holidays.csv"
        client = _RecordingClient(
            periods=[
                _summer(),
                SchoolHolidayPeriod(start_date=date(2025, 10, 20), end_date=date(2025, 10, 24), name="Autumn"),
            ]
        )
        transform = SchoolHolidayFeatureAdder(country_code="NL", subdivision_code="NL-NH", client=client)

        # Act
        transform.export_csv(path, start=date(2025, 1, 1), end=date(2025, 12, 31))

        # Assert
        assert len(client.requests) == 1
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        assert list(frame.columns) == [
            "country_code",
            "subdivision_code",
            "covered_from",
            "covered_until",
            "start_date",
            "end_date",
            "name",
            "subdivisions",
            "group_code",
        ]
        assert frame["covered_from"].tolist() == ["2025-01-01", "2025-01-01"]
        assert frame["covered_until"].tolist() == ["2025-12-31", "2025-12-31"]
        assert frame["name"].tolist() == ["Summer Holidays", "Autumn"]

    def test_export_overwrites_existing_file(self, tmp_path: Path) -> None:
        # Arrange
        path = tmp_path / "holidays.csv"
        path.write_text("stale content\n", encoding="utf-8")
        transform = SchoolHolidayFeatureAdder(country_code="NL", client=_RecordingClient(periods=[_summer()]))

        # Act
        transform.export_csv(path, start=date(2025, 1, 1), end=date(2025, 12, 31))

        # Assert
        assert "stale content" not in path.read_text(encoding="utf-8")

    def test_export_rejects_end_before_start(self, tmp_path: Path) -> None:
        # Arrange
        client = _RecordingClient()
        transform = SchoolHolidayFeatureAdder(country_code="NL", client=client)

        # Act / Assert
        with pytest.raises(ValueError, match="must not be before"):
            transform.export_csv(tmp_path / "holidays.csv", start=date(2025, 12, 31), end=date(2025, 1, 1))
        assert client.requests == []

    @pytest.mark.parametrize(
        "subdivision_code", [pytest.param("NL-NH", id="regional"), pytest.param(None, id="all-regions")]
    )
    def test_round_trip_restores_fitted_transform(self, tmp_path: Path, subdivision_code: str | None) -> None:
        # Arrange
        path = tmp_path / "holidays.csv"
        client = _RecordingClient(periods=[_summer()])
        SchoolHolidayFeatureAdder(country_code="NL", subdivision_code=subdivision_code, client=client).export_csv(
            path, start=date(2025, 1, 1), end=date(2025, 12, 31)
        )

        # Act
        with patch("requests.Session.send", side_effect=AssertionError("network access")):
            loaded = SchoolHolidayFeatureAdder.from_csv(
                path, country_code=CountryAlpha2("NL"), subdivision_code=subdivision_code
            )
            result = loaded.transform(_dataset("2025-06-30", periods=5))

        # Assert
        assert loaded.is_fitted
        assert loaded.subdivision_code == subdivision_code
        assert loaded.covered_from == date(2025, 1, 1)
        assert loaded.covered_until == date(2025, 12, 31)
        assert loaded.holidays == [_nationwide(_summer())]
        assert result.data["is_school_holiday"].tolist() == [0, 1, 1, 1, 0]

    def test_one_hot_round_trip_restores_name_columns(self, tmp_path: Path) -> None:
        # Arrange
        path = tmp_path / "holidays.csv"
        client = _RecordingClient(periods=[_summer(), _may()])
        SchoolHolidayFeatureAdder(country_code="NL", client=client).export_csv(
            path, start=date(2025, 1, 1), end=date(2025, 12, 31)
        )

        # Act
        with patch("requests.Session.send", side_effect=AssertionError("network access")):
            loaded = SchoolHolidayFeatureAdder.from_csv(path, country_code=CountryAlpha2("NL"), one_hot_by_name=True)
            result = loaded.transform(_dataset("2025-04-30", periods=5))

        # Assert
        assert loaded.features_added() == [
            "is_school_holiday",
            "is_school_holiday_may_holidays",
            "is_school_holiday_summer_holidays",
        ]
        assert result.data["is_school_holiday_may_holidays"].tolist() == [0, 1, 1, 0, 0]

    def test_one_hot_round_trip_keeps_columns_for_window_without_holidays(self, tmp_path: Path) -> None:
        # Arrange
        path = tmp_path / "holidays.csv"
        client = _RecordingClient(periods=[_summer(), _may()])
        SchoolHolidayFeatureAdder(country_code="NL", client=client).export_csv(
            path, start=date(2025, 1, 1), end=date(2025, 12, 31)
        )
        loaded = SchoolHolidayFeatureAdder.from_csv(path, country_code=CountryAlpha2("NL"), one_hot_by_name=True)

        # Act
        restored = pickle.loads(pickle.dumps(loaded))  # noqa: S301
        result = restored.transform(_dataset("2025-01-01", periods=3))

        # Assert
        assert restored.features_added() == [
            "is_school_holiday",
            "is_school_holiday_may_holidays",
            "is_school_holiday_summer_holidays",
        ]
        assert result.data["is_school_holiday_may_holidays"].tolist() == [0, 0, 0]
        assert result.data["is_school_holiday_summer_holidays"].tolist() == [0, 0, 0]

    @pytest.mark.parametrize(
        ("export_subdivision", "load_country", "load_subdivision"),
        [
            pytest.param("NL-NH", "DE", "NL-NH", id="wrong-country"),
            pytest.param("NL-NH", "NL", None, id="missing-subdivision"),
            pytest.param(None, "NL", "NL-NH", id="unexpected-subdivision"),
        ],
    )
    def test_from_csv_rejects_different_region(
        self, tmp_path: Path, export_subdivision: str | None, load_country: str, load_subdivision: str | None
    ) -> None:
        # Arrange
        path = tmp_path / "holidays.csv"
        SchoolHolidayFeatureAdder(
            country_code="NL", subdivision_code=export_subdivision, client=_RecordingClient(periods=[_summer()])
        ).export_csv(path, start=date(2025, 1, 1), end=date(2025, 12, 31))

        # Act / Assert
        with pytest.raises(ConfigurationError, match="was exported for"):
            SchoolHolidayFeatureAdder.from_csv(
                path, country_code=CountryAlpha2(load_country), subdivision_code=load_subdivision
            )

    def test_export_without_holidays_loads_empty_fitted_transform(self, tmp_path: Path) -> None:
        # Arrange
        path = tmp_path / "holidays.csv"
        SchoolHolidayFeatureAdder(country_code="NL", client=_RecordingClient()).export_csv(
            path, start=date(2025, 1, 1), end=date(2025, 1, 31)
        )

        # Act
        with patch("requests.Session.send", side_effect=AssertionError("network access")):
            loaded = SchoolHolidayFeatureAdder.from_csv(path, country_code=CountryAlpha2("NL"))
            result = loaded.transform(_dataset("2025-01-15", periods=3))

        # Assert
        assert loaded.is_fitted
        assert loaded.covered_from == date(2025, 1, 1)
        assert loaded.covered_until == date(2025, 1, 31)
        assert loaded.holidays == []
        assert result.data["is_school_holiday"].tolist() == [0, 0, 0]

    def test_from_csv_header_only_file_is_fitted_without_holidays(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Arrange
        path = tmp_path / "holidays.csv"
        path.write_text("country_code,subdivision_code,covered_from,covered_until,start_date,end_date,name\n")

        # Act
        loaded = SchoolHolidayFeatureAdder.from_csv(path, country_code=CountryAlpha2("NL"))
        with caplog.at_level(logging.WARNING):
            result = loaded.transform(_dataset("2025-01-15", periods=3))

        # Assert
        assert loaded.is_fitted
        assert loaded.covered_from is None
        assert loaded.holidays == []
        assert result.data["is_school_holiday"].tolist() == [0, 0, 0]
        assert "outside the school holiday coverage" not in caplog.text

    def test_from_csv_zero_byte_file_is_fitted_without_holidays(self, tmp_path: Path) -> None:
        # Arrange
        path = tmp_path / "holidays.csv"
        path.write_text("")

        # Act
        loaded = SchoolHolidayFeatureAdder.from_csv(path, country_code=CountryAlpha2("NL"))

        # Assert
        assert loaded.is_fitted
        assert loaded.holidays == []

    def test_from_csv_rejects_partially_filled_holiday_row(self, tmp_path: Path) -> None:
        # Arrange
        path = tmp_path / "holidays.csv"
        path.write_text(
            "country_code,subdivision_code,covered_from,covered_until,start_date,end_date,name\n"
            "NL,,2025-01-01,2025-12-31,2025-07-01,,Summer Holidays\n",
            encoding="utf-8",
        )

        # Act / Assert
        with pytest.raises(ConfigurationError, match="not a school holiday export"):
            SchoolHolidayFeatureAdder.from_csv(path, country_code=CountryAlpha2("NL"))

    def test_from_csv_rejects_missing_columns(self, tmp_path: Path) -> None:
        # Arrange
        path = tmp_path / "holidays.csv"
        path.write_text("country_code,start_date\nNL,2025-07-01\n", encoding="utf-8")

        # Act / Assert
        with pytest.raises(ConfigurationError, match="not a school holiday export"):
            SchoolHolidayFeatureAdder.from_csv(path, country_code=CountryAlpha2("NL"))

    def test_fit_reads_csv_path_without_contacting_client(self, tmp_path: Path) -> None:
        # Arrange
        path = tmp_path / "holidays.csv"
        SchoolHolidayFeatureAdder(country_code="NL", client=_RecordingClient(periods=[_summer()])).export_csv(
            path, start=date(2025, 1, 1), end=date(2025, 12, 31)
        )
        client = _RecordingClient()
        transform = SchoolHolidayFeatureAdder(country_code="NL", csv_path=path, client=client)

        # Act
        transform.fit(_dataset("2025-01-01", periods=10))

        # Assert
        assert client.requests == []
        assert transform.is_fitted
        assert transform.covered_until == date(2025, 12, 31)
        assert transform.holidays == [_nationwide(_summer())]

    def test_fit_csv_path_rejects_other_country(self, tmp_path: Path) -> None:
        # Arrange
        path = tmp_path / "holidays.csv"
        SchoolHolidayFeatureAdder(country_code="NL", client=_RecordingClient(periods=[_summer()])).export_csv(
            path, start=date(2025, 1, 1), end=date(2025, 12, 31)
        )
        transform = SchoolHolidayFeatureAdder(country_code="DE", csv_path=path)

        # Act / Assert
        with pytest.raises(ConfigurationError, match="was exported for"):
            transform.fit(_dataset("2025-01-01", periods=10))
        assert not transform.is_fitted

    def test_checkpoint_round_trip_keeps_fitted_holidays(self, tmp_path: Path) -> None:
        # Arrange
        path = tmp_path / "holidays.csv"
        SchoolHolidayFeatureAdder(country_code="NL", client=_RecordingClient(periods=[_summer()])).export_csv(
            path, start=date(2025, 1, 1), end=date(2025, 12, 31)
        )
        transform = SchoolHolidayFeatureAdder(country_code="NL", csv_path=path)
        transform.fit(_dataset("2025-01-01", periods=10))

        # Act
        restored = pickle.loads(pickle.dumps(transform))  # noqa: S301

        # Assert
        assert restored.is_fitted
        assert restored.covered_from == date(2025, 1, 1)
        assert restored.covered_until == date(2025, 12, 31)
        assert restored.holidays == [_nationwide(_summer())]


def _autumn(start: date, end: date, *codes: str) -> SchoolHolidayPeriod:
    return SchoolHolidayPeriod(start_date=start, end_date=end, name="Autumn", subdivisions=list(codes))


def _ge_holiday() -> SchoolHolidayPeriod:
    return _autumn(date(2025, 10, 20), date(2025, 10, 24), "NL-GE", "NL-GE-AR")


def _dr_holiday() -> SchoolHolidayPeriod:
    return _autumn(date(2025, 10, 27), date(2025, 10, 31), "NL-DR")


class TestRegionalGroups:
    """Tests for the opt-in regional_groups mode: one 0/1 column per regional holiday group."""

    def test_adds_one_column_per_group_and_no_plain_column(self) -> None:
        # Arrange
        client = _RecordingClient(periods=[_ge_holiday(), _dr_holiday()])
        transform = SchoolHolidayFeatureAdder(country_code="NL", regional_groups=True, client=client)
        transform.fit(_dataset("2025-10-18", periods=15))

        # Act
        result = transform.transform(_dataset("2025-10-18", periods=15)).data

        # Assert
        assert "is_school_holiday" not in result.columns
        # NL-DR sorts before NL-GE, so it is group 0 and NL-GE is group 1.
        assert result["is_school_holiday_1"].tolist() == [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        assert result["is_school_holiday_0"].tolist() == [0] * 9 + [1] * 5 + [0]

    def test_holiday_without_subdivisions_gets_nationwide_column(self) -> None:
        # Arrange
        client = _RecordingClient(periods=[_summer()])
        transform = SchoolHolidayFeatureAdder(country_code="NL", regional_groups=True, client=client)
        transform.fit(_dataset("2025-06-30", periods=5))

        # Act
        result = transform.transform(_dataset("2025-06-30", periods=5)).data

        # Assert
        assert transform.features_added() == ["is_school_holiday_0"]
        assert result["is_school_holiday_0"].tolist() == [0, 1, 1, 1, 0]

    def test_records_of_the_same_holiday_merge_into_one_group(self) -> None:
        # Arrange
        client = _RecordingClient(
            periods=[
                _autumn(date(2025, 10, 20), date(2025, 10, 24), "NL-GE"),
                _autumn(date(2025, 10, 20), date(2025, 10, 24), "NL-DR"),
            ]
        )
        transform = SchoolHolidayFeatureAdder(country_code="NL", regional_groups=True, client=client)
        transform.fit(_dataset("2025-10-18", periods=15))

        # Act
        result = transform.transform(_dataset("2025-10-18", periods=15)).data

        # Assert
        group_columns = [column for column in result.columns if column.startswith("is_school_holiday_")]
        assert group_columns == ["is_school_holiday_0"]
        assert result["is_school_holiday_0"].tolist() == [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    def test_holidays_covering_the_same_subdivisions_share_a_group(self) -> None:
        # Arrange
        client = _RecordingClient(
            periods=[
                _autumn(date(2025, 10, 20), date(2025, 10, 24), "NL-GE", "NL-GE-AR"),
                _autumn(date(2025, 10, 27), date(2025, 10, 31), "NL-GE", "NL-GE-AR"),
            ]
        )
        transform = SchoolHolidayFeatureAdder(country_code="NL", regional_groups=True, client=client)
        transform.fit(_dataset("2025-10-18", periods=15))

        # Act
        result = transform.transform(_dataset("2025-10-18", periods=15)).data

        # Assert
        assert transform.features_added() == ["is_school_holiday_0"]
        assert result["is_school_holiday_0"].tolist() == [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0]

    def test_groups_are_numbered_by_sorted_subdivisions_not_api_order(self) -> None:
        # Arrange
        client = _RecordingClient(periods=[_dr_holiday(), _ge_holiday()])
        transform = SchoolHolidayFeatureAdder(country_code="NL", regional_groups=True, client=client)

        # Act
        transform.fit(_dataset("2025-10-18", periods=15))

        # Assert
        assert transform.features_added() == ["is_school_holiday_0", "is_school_holiday_1"]
        assert [(period.group_code, period.subdivisions) for period in transform.holidays] == [
            ("0", ["NL-DR"]),
            ("1", ["NL-GE", "NL-GE-AR"]),
        ]

    def test_group_membership_is_deducible_from_fitted_holidays(self) -> None:
        # Arrange
        client = _RecordingClient(periods=[_ge_holiday(), _dr_holiday()])
        transform = SchoolHolidayFeatureAdder(country_code="NL", regional_groups=True, client=client)

        # Act
        transform.fit(_dataset("2025-10-18", periods=15))

        # Assert
        assert transform._group_members == {"0": frozenset({"NL-DR"}), "1": frozenset({"NL-GE", "NL-GE-AR"})}

    def test_group_columns_stay_fixed_after_fit(self) -> None:
        # Arrange
        client = _RecordingClient(periods=[_ge_holiday(), _dr_holiday()])
        transform = SchoolHolidayFeatureAdder(country_code="NL", regional_groups=True, client=client)
        transform.fit(_dataset("2025-10-18", periods=15))

        # Act
        result = transform.transform(_dataset("2030-01-01", periods=5)).data

        # Assert
        group_columns = {column for column in result.columns if column.startswith("is_school_holiday_")}
        assert group_columns == {"is_school_holiday_0", "is_school_holiday_1"}
        assert not result[sorted(group_columns)].any().any()

    def test_csv_round_trip_keeps_group_columns(self, tmp_path: Path) -> None:
        # Arrange
        path = tmp_path / "holidays.csv"
        periods = [_ge_holiday(), _dr_holiday()]
        SchoolHolidayFeatureAdder(
            country_code="NL", regional_groups=True, client=_RecordingClient(periods=periods)
        ).export_csv(path, start=date(2025, 1, 1), end=date(2025, 12, 31))
        fitted = SchoolHolidayFeatureAdder(
            country_code="NL", regional_groups=True, client=_RecordingClient(periods=periods)
        )
        fitted.fit(_dataset("2025-10-18", periods=15))

        # Act
        with patch("requests.Session.send", side_effect=AssertionError("network access")):
            loaded = SchoolHolidayFeatureAdder.from_csv(path, country_code=CountryAlpha2("NL"), regional_groups=True)
            result = loaded.transform(_dataset("2025-10-18", periods=15))

        # Assert
        pd.testing.assert_frame_equal(result.data, fitted.transform(_dataset("2025-10-18", periods=15)).data)

    def test_regional_load_rejects_csv_without_group_codes(self, tmp_path: Path) -> None:
        # Arrange
        path = tmp_path / "holidays.csv"
        path.write_text(
            "country_code,subdivision_code,covered_from,covered_until,start_date,end_date,name\n"
            "NL,,2025-01-01,2025-12-31,2025-07-01,2025-07-03,Summer Holidays\n"
        )

        # Act / Assert
        with pytest.raises(ConfigurationError, match="exported before regional groups"):
            SchoolHolidayFeatureAdder.from_csv(path, country_code=CountryAlpha2("NL"), regional_groups=True)
        plain = SchoolHolidayFeatureAdder.from_csv(path, country_code=CountryAlpha2("NL"))
        assert plain.holidays == [_summer()]

    def test_regional_load_rejects_csv_with_name_based_group_codes(self, tmp_path: Path) -> None:
        # Arrange
        path = tmp_path / "holidays.csv"
        path.write_text(
            "country_code,subdivision_code,covered_from,covered_until,start_date,end_date,name,subdivisions,group_code\n"
            "NL,,2025-01-01,2025-12-31,2025-10-20,2025-10-24,Autumn,NL-GE;NL-GE-AR,NL-GE\n"
        )

        # Act / Assert
        with pytest.raises(ConfigurationError, match="exported before regional groups"):
            SchoolHolidayFeatureAdder.from_csv(path, country_code=CountryAlpha2("NL"), regional_groups=True)

    def test_rejects_subdivision_code_with_regional_groups(self) -> None:
        with pytest.raises(ValidationError, match="subdivision_code must be None"):
            SchoolHolidayFeatureAdder(country_code="NL", subdivision_code="NL-NH", regional_groups=True)

    def test_config_rejects_subdivision_code_with_regional_groups(self) -> None:
        with pytest.raises(ValidationError, match="subdivision_code must be None"):
            SchoolHolidayConfig(country_code=CountryAlpha2("NL"), subdivision_code="NL-NH", regional_groups=True)
