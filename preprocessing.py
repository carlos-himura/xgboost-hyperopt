import pandas as pd

def preprocess_data(df: pd.DataFrame, is_training: bool = True):
    """
    Apply the same feature engineering and feature selection steps
    used during model training.

    Parameters:
        df (pd.DataFrame): Input raw dataframe (with original features).
        is_training (bool): Whether the data includes the 'target' column.

    Returns:
        pd.DataFrame: Processed dataframe ready for model input.
    """

    # --- Feature Engineering ---
    if all(col in df.columns for col in ['feature_2', 'feature_9', 'feature_13']):
        df['feature_2_9_13'] = df['feature_2'] * df['feature_9'] * df['feature_13']
        df['feature_9_x_13'] = df['feature_9'] * df['feature_13']
    else:
        raise ValueError("❌ Missing one or more required columns: 'feature_2', 'feature_9', 'feature_13'")

    # --- Select only relevant features ---
    selected_features = [
        'feature_2_9_13',
        'feature_9_x_13',
        'feature_11',
        'feature_18',
        'feature_2'
    ]

    # Include target only if this is for training
    if is_training:
        selected_features.append('target')

    # Filter the dataset
    missing_features = [col for col in selected_features if col not in df.columns]
    if missing_features:
        raise ValueError(f"❌ Missing expected columns in the input dataset: {missing_features}")

    df_processed = df[selected_features].copy()

    return df_processed