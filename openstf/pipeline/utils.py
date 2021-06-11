from datetime import datetime, timedelta, timezone


def generate_forecast_datetime_range(
    resolution_minutes: int, horizon_minutes: int
) -> tuple[datetime, datetime]:
    # get current date and time UTC
    datetime_utc = datetime.now(timezone.utc)
    # Datetime range for time interval to be predicted
    forecast_start = datetime_utc - timedelta(minutes=resolution_minutes)
    forecast_end = datetime_utc + timedelta(minutes=horizon_minutes)

    return forecast_start, forecast_end
