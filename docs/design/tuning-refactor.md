<!--
SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>

SPDX-License-Identifier: MPL-2.0
-->

# Tuning Module Refactoring — Design Doc

> **Branch:** `feature/684-openstef-40-support-hyperparameter-tuning`
> **Status:** Rev 4 (expanded)

---

## 1. Problem Statement

[`openstef_models/utils/tuning.py`](../../packages/openstef-models/src/openstef_models/utils/tuning.py) (665 lines) mixes Optuna-independent data structures / search-space logic with Optuna-specific orchestration in a single file. `import optuna` sits at module top level, so even importing `FloatRange` pulls in an optional dependency. `TunableHyperParams` exists solely to add a validator and private attr that should live on `HyperParams` itself. `TuningConfigMixin` duplicates what `BaseConfig` could provide directly.

---

## 2. Goals

1. **Split by responsibility** — range types and search-space extraction in core, Optuna orchestration in models.
2. **Eliminate `TunableHyperParams`** — fold into `HyperParams`.
3. **Eliminate `TuningConfigMixin`** — fold into `BaseConfig`.
4. **Ranges own their resolution** — `resolve()` method on each range type, replacing standalone `_merge_*` functions.
5. **Pydantic-native ranges** — use frozen Pydantic models (consistent with codebase; gains validation).
6. **Optuna stays optional** — lazy-imported only inside functions that need it.
7. **Bundle Optuna functions** — group into `HyperparameterTuner` class; eliminate loose functions.
8. **Testable in isolation** — core layer testable without Optuna.

---

## 3. Architecture

```
openstef-core (no optuna, no new deps)
┌─────────────────────────────────────┐
│ param_ranges.py                     │
│  FloatRange / IntRange / CatRange   │
│   + resolve(class_default) method   │
│  TuningRange (type alias)           │
│  ModelTuningInfo (frozen pydantic)  │
│                                     │
│ mixins/predictor.py — HyperParams   │
│  + _instance_ranges (PrivateAttr)   │
│  + _extract_tuning_ranges (valid.)  │
│  + get_search_space(include)        │
│                                     │
│ base_model.py — BaseConfig          │
│  + get_model_tuning_info()          │
└─────────────────────────────────────┘
              ▲
openstef-models (optuna lazy)
┌─────────────────────────────────────┐
│ utils/tuning.py                     │
│  HyperparameterTuner class          │
│   tune() / fit_with_tuning()        │
│   _objective / _suggest / etc.      │
│  TuningResult                       │
│  run_optuna_study()                 │
│  apply_trial_suggestions()          │
└─────────────────────────────────────┘
```

---

## 4. Key Decisions

### 4.1 Range types → frozen Pydantic models in `openstef_core/param_ranges.py`

**New file.** Each range is a frozen Pydantic model with a `resolve(class_default)` method that fills `None` bounds from the class-level default value. This eliminates the three standalone `_merge_*` functions and the `isinstance` dispatch in the current code.

Pydantic gives us validation for free — e.g. a `model_validator` can enforce `low <= high` on `FloatRange`/`IntRange`, or reject empty choice lists on `CategoricalRange`.

```python
class FloatRange(BaseModel):
    """Annotate a HyperParams float field as tunable within [low, high]."""
    model_config = ConfigDict(frozen=True)

    low: float | None = None
    high: float | None = None
    log: bool = False

    def resolve(self, class_default: float) -> FloatRange:
        """Fill None bounds from the class-level default."""
        return self.model_copy(update={
            "low": self.low if self.low is not None else class_default * 0.1,
            "high": self.high if self.high is not None else class_default * 10.0,
        })
```

`IntRange` is analogous. `CategoricalRange` holds `choices: list[Any]` and its `resolve()` is the identity (no bounds to fill).

**`TuningRange`** is a type alias: `type TuningRange = FloatRange | IntRange | CategoricalRange`.

**`ModelTuningInfo`** also lives here — a frozen Pydantic model holding `(field_name: str, hyperparams: HyperParams)`. Pure data, no Optuna dependency. Currently named `ModelTuningInfo` with `model_hyperparams_field_name` / `tunable_hyperparams` — renamed for brevity.

### 4.2 `HyperParams` gains tuning awareness

Currently, `TunableHyperParams(HyperParams)` adds:
1. A `model_validator(mode="wrap")` that intercepts `TuningRange` values passed as constructor kwargs (e.g. `learning_rate=FloatRange(0.01, 1.0)`) and stores them in a private attr before replacing them with the field default.
2. A `get_search_space()` method that returns `dict[str, TuningRange]`, resolving defaults from the class definition.

Both move into `HyperParams` itself. The validator logic is identical — it scans `__init__` kwargs for `TuningRange` instances, pops them into `_instance_ranges: dict[str, TuningRange]` (a `PrivateAttr`), and lets the remaining kwargs flow through to normal Pydantic validation.

`get_search_space(include: set[str] | None = None)` merges `_instance_ranges` over the class-level `Annotated[..., FloatRange(...)]` metadata, calls `range.resolve(class_default)` on each, and optionally filters by `include`.

**Effect on subclasses:** `XGBoostHyperParams`, `GBLinearHyperParams`, and any future forecaster params just inherit `HyperParams` directly (no more `TunableHyperParams`). Their field annotations stay identical:

```python
class XGBoostHyperParams(HyperParams):
    learning_rate: Annotated[float, FloatRange(low=0.01, high=1.0)] = 0.3
    max_depth: Annotated[int, IntRange(low=1, high=15)] = 6
    ...
```

Imports change from `openstef_models.utils.tuning` to `openstef_core.param_ranges`.

### 4.3 `BaseConfig` gains `get_model_tuning_info()`

Currently `TuningConfigMixin` provides this method — it introspects the config's fields for any that are `HyperParams` subclasses, and returns `list[ModelTuningInfo]`. After the move, every `BaseConfig` subclass inherits it.

Implementation uses a lazy import of `HyperParams` inside the method body to avoid a circular dependency (`base_model.py` ← `mixins/predictor.py`). Returns `[]` when no `HyperParams`-typed fields exist (the common case for non-forecasting configs).

`ForecastingWorkflowConfig` and `EnsembleForecastingWorkflowConfig` drop the `TuningConfigMixin` base — they just inherit `BaseConfig` and the method is already there.

### 4.4 `HyperparameterTuner` bundles Optuna logic

Everything Optuna-specific lives in a single class in `openstef_models/utils/tuning.py`. The class lazy-imports `optuna` inside its methods — so importing the module is safe even without optuna installed (for type-checking, IDE support, etc.).

```python
class HyperparameterTuner[ConfigT: BaseConfig]:
    def __init__(
        self,
        config: ConfigT,
        train_dataset: TimeSeriesDataset,
        create_workflow: Callable[[ConfigT], CustomForecastingWorkflow],
        *,
        n_trials: int = 20,
        seed: int | None = 42,
    ) -> None: ...

    def tune(self) -> TuningResult[ConfigT]: ...
    def fit_with_tuning(self) -> TuningResult[ConfigT]: ...
```

Internal methods: `_build_objective()`, `_apply_trial_suggestions()`, `_run_study()`.

**Convenience functions** at module level delegate to the class for backward-compatible usage:

```python
def tune(config: ConfigT, ..., *, n_trials: int = 20, seed: int | None = 42) -> TuningResult[ConfigT]:
    return HyperparameterTuner(config, ..., n_trials=n_trials, seed=seed).tune()
```

`TuningResult` remains a frozen Pydantic model (or named tuple) holding the tuned config + study results.

`run_optuna_study()` stays as a standalone utility (it's a thin wrapper around `optuna.create_study` / `study.optimize` — useful for custom use cases).

### 4.5 `optuna_n_trials` / `optuna_seed` move to `HyperparameterTuner`

These are tuner configuration, not workflow configuration. Currently they live on `ForecastingWorkflowConfig` (via `TuningConfigMixin`) and `EnsembleForecastingWorkflowConfig`. After refactoring:

- **Removed from:** `ForecastingWorkflowConfig`, `EnsembleForecastingWorkflowConfig`
- **Added to:** `HyperparameterTuner.__init__()` as `n_trials` and `seed`
- **`model_selection_metric`** stays on workflow configs — it describes which metric the model is optimized for, which is workflow-level concern.

Call sites that currently read `config.optuna_n_trials` will instead pass the value when constructing the tuner.

### 4.6 Testability

| Layer | Optuna needed? |
|-------|----------------|
| Core: ranges (`resolve()`) | No |
| Core: `HyperParams` (validator, `get_search_space`) | No |
| Core: `BaseConfig.get_model_tuning_info()` | No |
| Models: `HyperparameterTuner` | Yes (mocked or real) |

---

## 5. Migration Checklist

### Phase 1: Core — range types and search-space extraction

- [ ] **Create `openstef_core/param_ranges.py`**
  - `FloatRange`, `IntRange`, `CategoricalRange` as frozen Pydantic models
  - `TuningRange` type alias
  - `ModelTuningInfo` frozen Pydantic model (rename fields for brevity)
  - `resolve()` method on each range
  - Validators: `low <= high` on numeric ranges, non-empty choices on categorical
  - `__all__` exports

- [ ] **Modify `HyperParams` in `mixins/predictor.py`**
  - Add `_instance_ranges: dict[str, TuningRange]` as `PrivateAttr(default_factory=dict)`
  - Add `_extract_tuning_ranges` `model_validator(mode="wrap")` — pops range instances from kwargs, stores in private attr
  - Add `get_search_space(include: set[str] | None = None) -> dict[str, TuningRange]` method
  - Import range types from `param_ranges`

- [ ] **Modify `BaseConfig` in `base_model.py`**
  - Add `get_model_tuning_info() -> list[ModelTuningInfo]` with lazy `HyperParams` import
  - Import `ModelTuningInfo` from `param_ranges`

- [ ] **Update core `__init__.py` / `__all__` exports**
  - Export `FloatRange`, `IntRange`, `CategoricalRange`, `TuningRange`, `ModelTuningInfo` from `param_ranges`
  - Re-export from `mixins/__init__.py` if needed

- [ ] **Tests for Phase 1** (no Optuna needed)
  - Range construction + validation + `resolve()`
  - `HyperParams` validator: ranges extracted from kwargs, defaults applied
  - `get_search_space()` with and without `include` filter
  - `BaseConfig.get_model_tuning_info()` introspection

### Phase 2: Models — rewrite tuning.py

- [ ] **Rewrite `openstef_models/utils/tuning.py`**
  - `HyperparameterTuner[ConfigT: BaseConfig]` class with `n_trials`, `seed` params
  - `tune()`, `fit_with_tuning()` methods (lazy `import optuna`)
  - Internal: `_build_objective()`, `_apply_trial_suggestions()`, `_run_study()`
  - `TuningResult` frozen Pydantic model
  - `run_optuna_study()` standalone utility
  - Module-level `tune()` / `fit_with_tuning()` convenience functions
  - `__all__` exports

- [ ] **Update forecaster HyperParams classes**
  - `xgboost_forecaster.py`: `XGBoostHyperParams(HyperParams)` — import ranges from `openstef_core.param_ranges`
  - `gblinear_forecaster.py`: `GBLinearHyperParams(HyperParams)` — same

- [ ] **Update workflow configs**
  - `ForecastingWorkflowConfig`: drop `TuningConfigMixin`, remove `optuna_n_trials` / `optuna_seed`
  - `EnsembleForecastingWorkflowConfig`: same
  - Update call sites to pass `n_trials` / `seed` to `HyperparameterTuner` directly

- [ ] **Delete dead code**
  - `TunableHyperParams` class
  - `TuningConfigMixin` class
  - `TunableWorkflowConfig` Protocol
  - Standalone `_merge_*` functions

- [ ] **Tests for Phase 2**
  - `HyperparameterTuner` with mocked Optuna study
  - End-to-end: config → tuner → tuned config
  - Verify `import openstef_models.utils.tuning` works without Optuna installed

### Phase 3: Validate

- [ ] `uv run poe all` (lint → format → type → cover → report)
- [ ] Verify `from openstef_core.param_ranges import FloatRange` works standalone
- [ ] Verify `import openstef_models.utils.tuning` does NOT import optuna at module level

---

## 6. Resolved Questions

| Question | Decision |
|----------|----------|
| Module naming | `param_ranges.py` — new top-level module in `openstef_core` |
| Re-exports | No re-exports. Hard-cut to new import paths. |
| `optuna_n_trials` / `optuna_seed` | Move to `HyperparameterTuner.__init__()` params |
