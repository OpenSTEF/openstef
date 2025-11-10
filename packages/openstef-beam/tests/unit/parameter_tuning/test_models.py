from openstef_beam.parameter_tuning.models import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
    ParameterSpace,
    LGBMParameterSpace,
)


def test_parameter_space_init():
    """Test that ParameterSpace can be initialized with valid Distribution instances."""

    param_space = ParameterSpace(
        learning_rate=FloatDistribution(low=0.01, high=0.1),  # type: ignore
        n_estimators=IntDistribution(low=50, high=200),  # type: ignore
        booster=CategoricalDistribution(choices=["gbtree", "dart"]),  # type: ignore
    )

    assert isinstance(param_space.learning_rate, FloatDistribution)  # type: ignore
    assert isinstance(param_space.n_estimators, IntDistribution)  # type: ignore
    assert isinstance(param_space.booster, CategoricalDistribution)  # type: ignore


def test_lgbm_parameter_space_preset():
    """Test that LGBMParameterSpace preset returns correct distributions."""

    lgbm_params = LGBMParameterSpace()

    assert isinstance(lgbm_params.learning_rate, FloatDistribution)
    assert isinstance(lgbm_params.num_leaves, IntDistribution)
    assert isinstance(lgbm_params.max_depth, IntDistribution)
    assert isinstance(lgbm_params.reg_lambda, FloatDistribution)
