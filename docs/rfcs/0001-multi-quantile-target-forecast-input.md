# RFC-0001: Multi-Quantile Target Support for ForecastInputDataset

- **Status**: Under Review
- **Created**: 2025-11-27
- **Authors**: @egordm
- **Tracking Issue**: N/A

## Summary

Extend `ForecastInputDataset` to support multiple target series as quantiles (e.g., P10, P50, P90). This enables training models directly on probabilistic targets from upstream forecasters.

## Motivation

We're building a **metaforecasting module** with:

1. **Stacking Forecaster**: Meta-predictor combining quantile outputs from multiple base forecasters
2. **Residual Forecaster**: Trained on residuals of another model to improve combined forecasts

Both need multi-quantile targets as "ground truth" for training.

### Why not train N separate models?

```python
# Inefficient approach
for quantile in [0.1, 0.5, 0.9]:
    model_q.fit(data_for_quantile_q)  # 3x training time
```

Problems:
- ~3x training time, redundant feature computation
- No shared learning across quantiles
- Code complexity managing multiple models

XGBoost supports multi-target training natively (`multi_strategy="one_output_per_tree"`). Multi-quantile targets enable efficient joint training in a single pass.

## Design

### API Changes

```python
dataset = ForecastInputDataset(data, target_column="load")

# New properties
dataset.has_quantile_targets  # bool: True if multi-quantile
dataset.target_quantiles      # list[Quantile]: e.g., [Q(0.1), Q(0.5), Q(0.9)]
dataset.target_quantiles_data # DataFrame with all quantile columns
dataset.primary_target_series # P50 if multi-quantile, else single target

# Backward compatible
dataset.target_series         # Alias for primary_target_series
```

### Column Naming

Pattern: `{target_column}_{quantile.format()}`

| Column | Meaning |
|--------|---------|
| `load` | Single target (legacy) |
| `load_quantile_P10` | 10th percentile target |
| `load_quantile_P50` | Median target |

### Detection

1. If `target_quantiles` param provided → use those
2. Else auto-detect columns matching `{target_column}_quantile_P*`
3. Else → single-target mode

### Validation

- Multi-quantile mode requires P50 (for `target_series` compatibility)
- All declared quantiles must have corresponding columns

## Design Decisions

### D1: Sample weights use primary target
Sample weights based on `primary_target_series`. Custom forecasters can override via `has_quantile_targets`.

### D2: Forecasters must validate support
Forecasters raise `InputValidationError` if they receive multi-quantile targets but don't support them. Fail-fast prevents silent bugs.

### D3: Evaluation uses primary target only
Always use `primary_target_series` (P50 for multi-quantile) as ground truth. Quantile-to-quantile metrics deferred to future work.

### D4: "Primary" not "median" terminology
Single-target datasets aren't necessarily medians (could be mean). `primary_target_series` avoids implying statistical meaning.

### D5: Target quantiles must match forecaster quantiles
If forecaster supports multi-quantile targets, `data.target_quantiles` must exactly match `forecaster.config.quantiles`. Prevents undefined training behavior.

## Drawbacks

- Adds conditional logic to dataset class
- New column naming convention to document
- Forecasters need updates to leverage feature

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|--------------|
| Separate `MultiQuantileTargetDataset` class | Union types everywhere, code duplication |
| Use `quantile_P*` naming (like ForecastDataset) | Conflicts with forecast output columns |

## Future Possibilities

- Multi-output training on all quantile targets simultaneously
- Quantile-to-quantile evaluation metrics
- Per-quantile scalers and sample weights
