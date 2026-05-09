def get_xy(df, feature_cols, target_col="goal/no goal"):
    X = df[feature_cols].copy()
    y = df[target_col]
    return X, y