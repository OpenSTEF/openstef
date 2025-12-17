import pytest
from pydantic import Field

from openstef_beam.evaluation.metric_providers import MetricProvider, RCRPSSampleWeightedProvider
from openstef_beam.parameter_tuning.models import (
    CategoricalDistribution,
    FloatDistribution,
    GBLinearParameterSpace,
    IntDistribution,
    LGBMLinearParameterSpace,
    LGBMParameterSpace,
    OptimizationMetric,
    ParameterSpace,
    XGBoostParameterSpace,
)
from openstef_core.mixins.predictor import HyperParams


def test_optimization_metric_init():
    """Test that OptimizationMetric can be initialized with valid metric provider and direction."""

    metric_provider = RCRPSSampleWeightedProvider(lower_quantile=0.01, upper_quantile=0.99)
    optimization_metric = OptimizationMetric(
        metric=metric_provider,
        direction_minimize=True,
    )

    assert optimization_metric.metric == metric_provider
    assert optimization_metric.direction_minimize is True


def test_optization_metric_unknown():
    """Test that OptimizationMetric raises ValueError for unknown metric provider."""

    class UnknownMetricProvider(MetricProvider):
        pass

    unknown_provider = UnknownMetricProvider()

    with pytest.raises(KeyError):
        OptimizationMetric(
            metric=unknown_provider,
            direction_minimize=True,
        )


def test_parameter_space_init():
    """Test that ParameterSpace can be initialized with valid Distribution instances."""

    class MockHyperParams(HyperParams):
        learning_rate: float = Field(default=0.1)
        n_estimators: int = Field(default=100)
        booster: str = Field(default="gbtree")

    class MockParameterSpace(ParameterSpace):
        learning_rate: FloatDistribution = Field(...)
        n_estimators: IntDistribution = Field(...)
        booster: CategoricalDistribution = Field(...)

        @classmethod
        def default_hyperparams(cls) -> MockHyperParams:
            return MockHyperParams()

    param_space = MockParameterSpace(
        learning_rate=FloatDistribution(low=0.01, high=0.1),
        n_estimators=IntDistribution(low=50, high=200),
        booster=CategoricalDistribution(choices=["gbtree", "dart"]),
    )

    assert isinstance(param_space.learning_rate, FloatDistribution)
    assert isinstance(param_space.n_estimators, IntDistribution)
    assert isinstance(param_space.booster, CategoricalDistribution)


def test_lgbm_parameter_space_preset():
    """Test that LGBMParameterSpace preset returns correct distributions."""

    lgbm_params = LGBMParameterSpace()

    assert isinstance(lgbm_params.learning_rate, FloatDistribution)
    assert isinstance(lgbm_params.num_leaves, IntDistribution)
    assert isinstance(lgbm_params.max_depth, IntDistribution)
    assert isinstance(lgbm_params.reg_lambda, FloatDistribution)


def test_lgbm_linear_parameter_space_preset():
    """Test that LGBMLinearParameterSpace preset returns correct distributions."""

    lgbm_linear_params = LGBMLinearParameterSpace()

    assert isinstance(lgbm_linear_params.learning_rate, FloatDistribution)
    assert isinstance(lgbm_linear_params.num_leaves, IntDistribution)
    assert isinstance(lgbm_linear_params.max_depth, IntDistribution)
    assert isinstance(lgbm_linear_params.reg_lambda, FloatDistribution)


def test_gblinear_parameter_space_preset():
    """Test that GBLinearParameterSpace preset returns correct distributions."""

    gblinear_params = GBLinearParameterSpace()

    assert isinstance(gblinear_params.learning_rate, FloatDistribution)
    assert isinstance(gblinear_params.reg_alpha, FloatDistribution)


def test_xgboost_parameter_space_preset():
    """Test that XGBoostParameterSpace preset returns correct distributions."""

    xgboost_params = XGBoostParameterSpace()

    assert isinstance(xgboost_params.learning_rate, FloatDistribution)
    assert isinstance(xgboost_params.max_depth, IntDistribution)
    assert isinstance(xgboost_params.subsample, FloatDistribution)
