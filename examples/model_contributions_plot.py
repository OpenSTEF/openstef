# SPDX-FileCopyrightText: 2017-2025 Contributors to the OpenSTEF project
# SPDX-License-Identifier: MPL-2.0
"""Model contribution visualization for ensemble forecasting.

This module provides functionality to visualize and analyze model contributions
in ensemble forecasting, specifically comparing GBLinear and LGBM base models
across different quantiles.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

from openstef_beam.analysis.plots import ForecastTimeSeriesPlotter

try:
    from openstef_beam.analysis.plots import ForecastTimeSeriesPlotter
except ImportError:
    ForecastTimeSeriesPlotter = None


def load_contribution_data(folder_path: Path, n_rows: int = 24) -> pd.DataFrame:
    """Load and concatenate contribution data from parquet files.

    Args:
        folder_path: Path to folder containing contribution parquet files.
        n_rows: Number of rows to take from each file (default: 24 for 6 hours
            at 15-minute intervals).

    Returns:
        Concatenated DataFrame with contribution data.

    Raises:
        ValueError: If no parquet files are found in the specified folder.
    """
    parquet_files = sorted(folder_path.glob("contrib_*_predict.parquet"))

    if not parquet_files:
        msg = f"No contribution files found in {folder_path}"
        raise ValueError(msg)

    print(f"Found {len(parquet_files)} contribution files")

    df_list = []
    for file in tqdm(parquet_files, desc="Loading contribution data"):
        df_temp = pd.read_parquet(file)
        df_subset = df_temp.head(n_rows)
        df_list.append(df_subset)

    df_combined = pd.concat(df_list, axis=0, ignore_index=False)
    print(f"Combined dataframe shape: {df_combined.shape}")
    if not df_combined.empty:
        start_ts = df_combined.index.min()
        end_ts = df_combined.index.max()
        print(f"Dataframe start: {start_ts}, end: {end_ts}")
    else:
        print("Combined dataframe is empty; no timestamps available.")

    return df_combined


def _filter_by_date_range(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Filter dataframe by date range, handling timezone-aware indices.

    Args:
        df: DataFrame with DatetimeIndex.
        start_date: Start date for filtering. If None, no start filtering.
        end_date: End date for filtering. If None, no end filtering.

    Returns:
        Filtered DataFrame.
    """
    df_filtered = df.copy()

    if start_date is not None:
        start_dt = pd.to_datetime(start_date)
        if df_filtered.index.tz is not None and start_dt.tz is None:
            start_dt = start_dt.tz_localize("UTC")
        df_filtered = df_filtered[df_filtered.index >= start_dt]

    if end_date is not None:
        end_dt = pd.to_datetime(end_date)
        if df_filtered.index.tz is not None and end_dt.tz is None:
            end_dt = end_dt.tz_localize("UTC")
        df_filtered = df_filtered[df_filtered.index <= end_dt]

    return df_filtered


def _create_model_traces(df: pd.DataFrame, quantiles: list[str], default_quantile: str = "P50") -> go.Figure:
    """Create Plotly traces for each model and quantile.

    Args:
        df: DataFrame with model contribution columns.
        quantiles: List of quantile names (e.g., ['P05', 'P50']).
        default_quantile: Quantile to show by default.

    Returns:
        Plotly Figure with traces for all models and quantiles.
    """
    fig = go.Figure()

    for quantile in quantiles:
        gblinear_col = f"gblinear_quantile_{quantile}"
        lgbm_col = f"lgbm_quantile_{quantile}"

        is_visible = quantile == default_quantile

        # Add GBLinear trace
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[gblinear_col],
                name=f"GBLinear - {quantile}",
                mode="lines",
                line={"width": 2, "color": "#1f77b4"},
                opacity=0.85,
                visible=is_visible,
            )
        )

        # Add LGBM trace
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[lgbm_col],
                name=f"LGBM - {quantile}",
                mode="lines",
                line={"width": 2, "color": "#ff7f0e"},
                opacity=0.85,
                visible=is_visible,
            )
        )

    return fig


def _create_quantile_buttons(quantiles: list[str]) -> list[dict]:
    """Create button controls for quantile selection.

    Args:
        quantiles: List of quantile names.

    Returns:
        List of button configuration dictionaries.
    """
    buttons = []
    for i, quantile in enumerate(quantiles):
        # Two traces per quantile (GBLinear and LGBM)
        visible = [False] * (len(quantiles) * 2)
        visible[i * 2] = True  # GBLinear
        visible[i * 2 + 1] = True  # LGBM

        buttons.append({
            "label": quantile,
            "method": "update",
            "args": [
                {"visible": visible},
                {"title.text": (f"Model Contributions - Quantile {quantile}")},
            ],
        })

    return buttons


def plot_model_contributions(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    quantiles: list[str] | None = None,
) -> go.Figure:
    """Plot model contributions with interactive quantile selection.

    Creates an interactive Plotly visualization comparing GBLinear and LGBM
    model contributions across different quantiles, with button controls to
    switch between quantiles.

    Args:
        df: DataFrame with model contribution columns. Expected columns:
            'gblinear_quantile_{Q}' and 'lgbm_quantile_{Q}' for each quantile.
        start_date: Start date for filtering. If None, uses all data.
        end_date: End date for filtering. If None, uses all data.
        quantiles: List of quantile names (e.g., ['P05', 'P50']). If None,
            defaults to ['P05', 'P10', 'P30', 'P50', 'P70', 'P90', 'P95'].

    Returns:
        Plotly Figure object with interactive visualization.

    Example:
        >>> df = load_contribution_data(Path("path/to/contributions"))
        >>> fig = plot_model_contributions(
        ...     df,
        ...     start_date='2024-03-01',
        ...     end_date='2024-03-31',
        ...     quantiles=['P50', 'P90']
        ... )
        >>> fig.show()
    """
    if quantiles is None:
        quantiles = ["P05", "P10", "P30", "P50", "P70", "P90", "P95"]

    # Filter data by date range
    df_plot = _filter_by_date_range(df, start_date, end_date)

    # Create traces for all models and quantiles
    default_quantile = "P50" if "P50" in quantiles else quantiles[0]
    fig = _create_model_traces(df_plot, quantiles, default_quantile)

    # Create quantile selection buttons
    buttons = _create_quantile_buttons(quantiles)
    active_idx = quantiles.index(default_quantile)

    # Update layout with controls and styling
    fig.update_layout(
        legend={
            "orientation": "v",
            "x": 1.02,
            "xanchor": "left",
            "y": 1.0,
            "yanchor": "top",
        },
        updatemenus=[
            {
                "type": "buttons",
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "active": active_idx,
                "x": 1.02,
                "xanchor": "left",
                "y": 0.5,
                "yanchor": "middle",
                "pad": {"r": 10, "t": 10},
                "bgcolor": "lightgray",
                "bordercolor": "gray",
                "borderwidth": 1,
            }
        ],
        title=f"Model Contributions - Quantile {quantiles[active_idx]}",
        xaxis_title="Timestamp",
        yaxis_title="Contribution",
        hovermode="x unified",
        height=600,
        showlegend=True,
    )

    return fig


def plot_combined_visualization(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    quantiles: list[str] | None = None,
) -> go.Figure:
    """Plot model contributions and forecast time series in subplots.

    Creates a combined visualization with two subplots:
    1. Top: Model contributions comparing GBLinear and LGBM
    2. Bottom: Forecast time series using ForecastTimeSeriesPlotter showing
       ensemble forecast, measurements, and uncertainty bands

    Args:
        df: DataFrame with model contribution columns, ensemble forecasts, and load.
            Expected columns: 'gblinear_quantile_{Q}', 'lgbm_quantile_{Q}',
            'quantile_{Q}', and 'load'.
        start_date: Start date for filtering. If None, uses all data.
        end_date: End date for filtering. If None, uses all data.
        quantiles: List of quantile names (e.g., ['P05', 'P50']). If None,
            defaults to ['P05', 'P10', 'P30', 'P50', 'P70', 'P90', 'P95'].

    Returns:
        Plotly Figure object with combined subplots.
    """
    if quantiles is None:
        quantiles = ["P05", "P10", "P30", "P50", "P70", "P90", "P95"]

    # Filter data by date range
    df_plot = _filter_by_date_range(df, start_date, end_date)

    # Create forecast plot using ForecastTimeSeriesPlotter
    # Prepare quantiles DataFrame for the plotter
    quantile_cols = [f"quantile_{q}" for q in quantiles]
    quantiles_df = df_plot[quantile_cols].copy()

    # Create measurements series
    measurements = df_plot["load"]

    # Use ForecastTimeSeriesPlotter to create the forecast visualization
    forecast_plotter = ForecastTimeSeriesPlotter()
    forecast_plotter.add_measurements(measurements)
    forecast_plotter.add_model(
        model_name="Ensemble",
        forecast=df_plot["quantile_P50"],
        quantiles=quantiles_df,
    )
    forecast_fig = forecast_plotter.plot(title="Ensemble Forecast vs Measurements")

    # Create subplot figure with 2 rows
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.5, 0.5],
        subplot_titles=("Ensemble Forecast vs Measurements", "Model Contributions"),
        vertical_spacing=0.12,
        shared_xaxes=True,
    )

    # Add forecast traces to top subplot (row=1)
    for trace in forecast_fig.data:
        fig.add_trace(trace, row=1, col=1)

    # Count the number of forecast traces for visibility calculation
    num_forecast_traces = len(forecast_fig.data)

    # Add contribution traces to bottom subplot (row=2)
    default_quantile = "P50" if "P50" in quantiles else quantiles[0]

    for quantile in quantiles:
        gblinear_col = f"gblinear_quantile_{quantile}"
        lgbm_col = f"lgbm_quantile_{quantile}"

        is_visible = quantile == default_quantile

        # Add GBLinear trace
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot[gblinear_col],
                name=f"GBLinear - {quantile}",
                mode="lines",
                line={"width": 2, "color": "#1f77b4"},
                opacity=0.85,
                visible=is_visible,
                legendgroup="contributions",
            ),
            row=2,
            col=1,
        )

        # Add LGBM trace
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot[lgbm_col],
                name=f"LGBM - {quantile}",
                mode="lines",
                line={"width": 2, "color": "#ff7f0e"},
                opacity=0.85,
                visible=is_visible,
                legendgroup="contributions",
            ),
            row=2,
            col=1,
        )

    # Create quantile selection buttons (only affects contribution subplot)
    buttons = []
    for i, quantile in enumerate(quantiles):
        # Calculate visibility for all traces
        # First set for forecast traces (always visible)
        visible = [True] * num_forecast_traces

        # Add visibility for contribution traces (2 traces per quantile)
        for j, _ in enumerate(quantiles):
            visible.extend([j == i, j == i])  # GBLinear and LGBM visibility

        buttons.append({
            "label": quantile,
            "method": "update",
            "args": [
                {"visible": visible},
                {"title.text": (f"Model Contributions & Forecast - Quantile {quantile}")},
            ],
        })

    active_idx = quantiles.index(default_quantile)

    # Update layout
    fig.update_layout(
        title=f"Model Contributions & Forecast - Quantile {quantiles[active_idx]}",
        height=1000,
        hovermode="x unified",
        showlegend=True,
        updatemenus=[
            {
                "type": "buttons",
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "active": active_idx,
                "x": 1.02,
                "xanchor": "left",
                "y": 0.5,
                "yanchor": "middle",
                "pad": {"r": 10, "t": 10},
                "bgcolor": "lightgray",
                "bordercolor": "gray",
                "borderwidth": 1,
            }
        ],
    )

    # Update axes labels
    fig.update_xaxes(title_text="Timestamp", row=2, col=1)
    fig.update_yaxes(title_text="Load (MW)", row=1, col=1)
    fig.update_yaxes(title_text="Contribution", row=2, col=1)

    return fig


def main() -> None:
    """Main function to demonstrate contribution visualization."""
    # Load contribution data
    # Get the project root (two levels up from this file)
    project_root = Path(__file__).parent.parent
    folder_path = (
        project_root
        / "benchmark_results"
        / "cache"
        / "Ensemble_contributions_lgbm_gblinear_learned_weights_lgbm_OS Apeldoorn"
    )

    df_combined = load_contribution_data(folder_path, n_rows=24)

    # Create combined visualization with contributions and forecast subplots
    combined_fig = plot_combined_visualization(df_combined)
    combined_output = project_root / "benchmark_results" / "model_contributions_combined_plot.html"
    combined_fig.write_html(combined_output)
    print(f"Combined plot saved to: {combined_output}")

    combined_fig.show()


if __name__ == "__main__":
    main()
