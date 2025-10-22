import argparse # Lets the Python script accepts command-line arguments — like --epochs 10
import os # Allow the interaction with the operating system: creating folders, reading env variables
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 
from preprocessing import preprocess_data  # Using the preprocessing script for feature engineering

def evaluate_model(model, X, y, dataset_name="Dataset"):
    preds = model.predict(xgb.DMatrix(X)) # Convert X into a DMatrix (XGBoost format), and use that DMatrix to run predictions efficiently. DMatrix is because we are not using scikit-learn 
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
    # Directories (on the container) that SageMaker maps from S3 channels
    train_path = os.path.join(args.train, "train.csv")
    valid_path = os.path.join(args.validation, "validation.csv")
    test_path = os.path.join(args.test, "test.csv")

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)

    # Apply preprocessing logic
    train_df = preprocess_data(train_df, is_training=True)
    valid_df = preprocess_data(valid_df, is_training=True)
    test_df  = preprocess_data(test_df, is_training=True)

    X_train, y_train = train_df.drop("target", axis=1), train_df["target"]
    X_valid, y_valid = valid_df.drop("target", axis=1), valid_df["target"]
    X_test,  y_test  = test_df.drop("target", axis=1), test_df["target"]

    # Converts your training data (X_train, y_train) into a DMatrix — the format XGBoost uses internally during training
    dtrain = xgb.DMatrix(X_train, label=y_train) 
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    dtest  = xgb.DMatrix(X_test, label=y_test)

    # --- Fixed Parameters (from your Hyperopt run) ---
    params = {
        'objective': 'reg:squarederror', # Defines what the model is trying to optimize during training
        'colsample_bytree': 0.7165837677011807, # Fraction of features (columns) sampled per tree. Adds randomness and prevents overfitting
        'gamma': 0.2848467602482611, # Minimum loss reduction required to make a further partition
        'learning_rate': 0.018394786480165966, # Step size shrinkage used in updates for Gradient Descent
        'max_depth': int(3.0), # Maximum depth of each tree. Controls model complexity
        'n_estimators': int(600.0), # Number of boosting rounds (trees). How many trees to build in total
        'reg_lambda': 0.4220839202817964, # L2 (Ridge reg) term on weights. Adds a penalty for large weights (in some trees), making the model more conservative. Don’t let leaf weights grow too large to avoid overfitting
        'subsample': 0.6808070045946493, # Fraction of training data used per tree. Helps prevent overfitting
        'eval_metric': 'rmse', # Defines how performance is measured during evaluation (validation/testing)
        'seed': 12345
    }

    print("--- Training XGBoost Model ---")
    evals = [(dtrain, 'train'), (dvalid, 'validation')] # Allows XGBoost to track performance and detect overfitting
    model = xgb.train(params, dtrain, num_boost_round=int(params['n_estimators']), evals=evals) # Native XGBoost training function. train.() is a Native XGBoost API

    # --- Evaluate ---
    evaluate_model(model, X_train, y_train, "Train")
    evaluate_model(model, X_valid, y_valid, "Validation")
    evaluate_model(model, X_test, y_test, "Test")

    # --- Save Model (Required Name for XGB Container) ---
    os.makedirs(args.model_dir, exist_ok=True) # Creates the directory where SageMaker expects the trained model to be saved
    output_path = os.path.join(args.model_dir, "xgboost-model") # Builds the full file path for the model file
    model.save_model(output_path) # Writes the trained XGBoost model to disk in binary format
    print(f"✅ Model saved to {output_path}")

if __name__ == "__main__": # Python convention that tells only run the code of this block if this file is being executed
    parser = argparse.ArgumentParser() # Creates a command-line argument parser using the argparse module
    # Environment variables automatically created by SageMaker inside the container
    # They tell the script (train.py) where to find the data and where to save the model
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    args = parser.parse_args() # Reads all the arguments and stores them in the args object
    main(args) # Calls the main training function, passing the parsed arguments