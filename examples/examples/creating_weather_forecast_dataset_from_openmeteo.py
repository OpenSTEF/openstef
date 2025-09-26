


from datetime import datetime
from pathlib import Path

from pydantic_extra_types.coordinate import Coordinate, Latitude, Longitude

from openstef_beam.benchmarks.data_sources.open_meteo import OpenMeteoDataRepository
from openstef_core.types import DatetimeRange

output_dir = Path(__file__).resolve().parent

repository = OpenMeteoDataRepository()


data = repository.fetch_weather_data(
    coordinate=Coordinate(latitude=Latitude(52.090737), longitude=Longitude(5.12142)),
    range=DatetimeRange(
        start=datetime.fromisoformat("2025-01-01T00:00:00+00:00"),
        end=datetime.fromisoformat("2025-01-08T00:00:00+00:00"),
    ),
)

print("Fetched weather data:")
print(data.data.head())

data.to_parquet(path=output_dir / "weather_data.parquet")


versioned_data = repository.fetch_weather_data_versioned(
    coordinate=Coordinate(latitude=Latitude(52.090737), longitude=Longitude(5.12142)),
    range=DatetimeRange(
        start=datetime.fromisoformat("2025-01-01T00:00:00+00:00"),
        end=datetime.fromisoformat("2025-01-08T00:00:00+00:00"),
    ),
)

print("Fetched versioned weather data:")
print(versioned_data.data_parts[0].data.head())

versioned_data.to_parquet(path=output_dir / "weather_data_versioned.parquet")
