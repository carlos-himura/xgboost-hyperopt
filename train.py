import argparse
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocessing import preprocess_data  # You can keep this if you use Script Mode

def evaluate_model(model, X, y, dataset_name="Dataset"):
    preds = model.predict(xgb.DMatrix(X))
    mae = mean_absolute_error(y, preds)
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, preds)
    print(f"📊 {dataset_name} Results:")
    print(f"   MAE  = {mae:.4f}")
    print(f"   MSE  = {mse:.4f}")
    print(f"   RMSE = {rmse:.4f}")
    print(f"   R²   = {r2:.4f}")
    print("-" * 40)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

def main(args):
    print("--- Loading Data ---")
    train_path = os.path.join(args.train, "train.csv")
    valid_path = os.path.join(args.validation, "validation.csv")
    test_path = os.path.join(args.test, "test.csv")

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)

    # Apply preprocessing logic (optional if needed)
    train_df = preprocess_data(train_df, is_training=True)
    valid_df = preprocess_data(valid_df, is_training=True)
    test_df  = preprocess_data(test_df, is_training=True)

    X_train, y_train = train_df.drop("target", axis=1), train_df["target"]
    X_valid, y_valid = valid_df.drop("target", axis=1), valid_df["target"]
    X_test,  y_test  = test_df.drop("target", axis=1), test_df["target"]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    dtest  = xgb.DMatrix(X_test, label=y_test)

    # --- Fixed Parameters (from your Hyperopt run) ---
    params = {
        'objective': 'reg:squarederror',
        'colsample_bytree': 0.7165837677011807,
        'gamma': 0.2848467602482611,
        'learning_rate': 0.018394786480165966,
        'max_depth': int(3.0),
        'n_estimators': int(600.0),
        'reg_lambda': 0.4220839202817964,
        'subsample': 0.6808070045946493,
        'eval_metric': 'rmse',
        'seed': 12345
    }

    print("--- Training XGBoost Model ---")
    evals = [(dtrain, 'train'), (dvalid, 'validation')]
    model = xgb.train(params, dtrain, num_boost_round=int(params['n_estimators']), evals=evals)

    # --- Evaluate ---
    evaluate_model(model, X_train, y_train, "Train")
    evaluate_model(model, X_valid, y_valid, "Validation")
    evaluate_model(model, X_test, y_test, "Test")

    # --- Save Model (Required Name for XGB Container) ---
    os.makedirs(args.model_dir, exist_ok=True)
    output_path = os.path.join(args.model_dir, "xgboost-model")
    model.save_model(output_path)
    print(f"✅ Model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    args = parser.parse_args()
    main(args)