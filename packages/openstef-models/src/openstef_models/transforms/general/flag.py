import pandas as pd

def compute_range_flag(df: pd.DataFrame,
                       input_features: pd.DataFrame | None, 
                       features_to_flag: list[str],
                       feature_ranges: dict[str, tuple[float, float]] | None = None
                       ):
    """Module for computing flags based on inside or outside feature ranges. 
    This function only applies to specified features"""
    if input_features is not None: 
        invalid_features = set(features_to_flag) - set(input_features.columns)
        if invalid_features:
            raise ValueError(f"Features to flag not found in input DataFrame: {invalid_features}")
    
    # Add inside_train_range flag (1 if all numeric input features are within training min/max)
    inside_flag = pd.Series(False, index=df.index)
    if input_features is not None and feature_ranges:
        # align input_features to prediction index where possible
        features_aligned = input_features.reindex(df.index)
        numeric = features_aligned.select_dtypes(include="number")
        if not numeric.empty:
            inside = pd.Series(True, index=numeric.index)
            for col in features_to_flag:
                mn, mx = feature_ranges.get(col, (None, None))
                if mn is not None and mx is not None:
                    # if inside of range: mask returns True
                    mask = numeric[col].ge(mn) & numeric[col].le(mx)
                    mask = mask.fillna(False) #type: ignore
                    inside &= mask
                else:
                    # missing range: treat as inside
                    continue
            inside_flag = inside.fillna(False) #type: ignore
    
    return inside_flag.astype(int)

