# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import plotly.graph_objects as go
import pytest

from openstef_beam.analysis.plots import QuantileProbabilityPlotter
from openstef_core.types import Q


class TestQuantileProbabilityPlotter:
    def test_add_model_returns_self(self):
        # Arrange
        plotter = QuantileProbabilityPlotter()

        # Act
        result = plotter.add_model("model1", [Q(0.1), Q(0.5)], [Q(0.2), Q(0.6)])

        # Assert
        assert result is plotter, "add_model should return self for method chaining"

    def test_add_model_appends_data(self):
        # Arrange
        plotter = QuantileProbabilityPlotter()

        # Act
        plotter.add_model("model1", [Q(0.1), Q(0.5)], [Q(0.2), Q(0.6)])

        # Assert
        assert len(plotter.models_data) == 1
        assert plotter.models_data[0]["model"] == "model1"
        assert plotter.models_data[0]["forecasted_prob"] == [Q(0.1), Q(0.5)]
        assert plotter.models_data[0]["observed_prob"] == [Q(0.2), Q(0.6)]

    def test_add_model_multiple_models(self):
        # Arrange
        plotter = QuantileProbabilityPlotter()

        # Act
        plotter.add_model("model1", [Q(0.1), Q(0.5)], [Q(0.2), Q(0.6)])
        plotter.add_model("model2", [Q(0.3), Q(0.7)], [Q(0.4), Q(0.8)])

        # Assert
        assert len(plotter.models_data) == 2
        assert plotter.models_data[0]["model"] == "model1"
        assert plotter.models_data[1]["model"] == "model2"

    @pytest.mark.parametrize(
        ("forecasted_probs", "observed_probs", "error_expected"),
        [
            pytest.param([Q(0.1), Q(0.5)], [Q(0.2), Q(0.6)], False, id="equal_lengths"),
            pytest.param([Q(0.1), Q(0.5), Q(0.7)], [Q(0.2), Q(0.6)], True, id="forecasted_longer"),
            pytest.param([Q(0.1), Q(0.5)], [Q(0.2), Q(0.6), Q(0.8)], True, id="observed_longer"),
        ],
    )
    def test_add_model_lengths_validation(
        self, forecasted_probs: list[Q], observed_probs: list[Q], error_expected: bool
    ):
        # Arrange
        plotter = QuantileProbabilityPlotter()

        # Act & Assert
        if error_expected:
            with pytest.raises(ValueError, match="must have the same length"):
                plotter.add_model("model1", forecasted_probs, observed_probs)
        else:
            plotter.add_model("model1", forecasted_probs, observed_probs)
            assert len(plotter.models_data) == 1

    def test_plot_empty_data_raises_error(self):
        # Arrange
        plotter = QuantileProbabilityPlotter()

        # Act & Assert
        with pytest.raises(ValueError, match="No model data has been added"):
            plotter.plot()

    def test_plot_returns_figure(self):
        # Arrange
        plotter = QuantileProbabilityPlotter()
        plotter.add_model("model1", [Q(0.1), Q(0.5)], [Q(0.2), Q(0.6)])

        # Act
        fig = plotter.plot()

        # Assert
        assert isinstance(fig, go.Figure)

    def test_plot_with_custom_title(self):
        # Arrange
        plotter = QuantileProbabilityPlotter()
        plotter.add_model("model1", [Q(0.1), Q(0.5)], [Q(0.2), Q(0.6)])
        custom_title = "Custom Plot Title"

        # Act
        fig = plotter.plot(title=custom_title)

        # Assert
        assert fig.layout.title.text == custom_title  # type: ignore

    def test_plot_with_multiple_models(self):
        # Arrange
        plotter = QuantileProbabilityPlotter()
        plotter.add_model("model1", [Q(0.1), Q(0.5)], [Q(0.2), Q(0.6)])
        plotter.add_model("model2", [Q(0.3), Q(0.7)], [Q(0.4), Q(0.8)])

        # Act
        fig = plotter.plot()

        # Assert
        # The figure should have 3 traces: 2 models + perfect calibration line
        assert len(fig.data) == 3  # type: ignore

        # Check that both models are represented in the plot
        model_names = [trace.name for trace in fig.data if trace.name != "Perfect probability"]  # type: ignore
        assert "model1" in model_names
        assert "model2" in model_names
