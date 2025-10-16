import argparse # Parse command-line arguments (SageMaker passes channel paths and model dir here)
import os # Path & environment variable handling
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold # used for cross-validation inside Hyperopt objective
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.pyll.base import scope
import joblib # serialize the trained model (joblib.dump)

# --- Evaluate Model (Metrics) ---
def evaluate_model(model, X, y, dataset_name="Dataset"):
    preds = model.predict(X)
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


# --- Main Training Function ---
def main(args):
    print("--- Loading Data ---")
    # Directories (on the container) that SageMaker maps from S3 channels
    train_path = os.path.join(args.train, "train.csv")
    valid_path = os.path.join(args.validation, "validation.csv")
    test_path = os.path.join(args.test, "test.csv")

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)

    # Separates features and target column
    X_train = train_df.drop("target", axis=1)
    y_train = train_df["target"]

    X_valid = valid_df.drop("target", axis=1)
    y_valid = valid_df["target"]

    X_test = test_df.drop("target", axis=1)
    y_test = test_df["target"]

    print(f"Train shape: {X_train.shape}, Validation: {X_valid.shape}, Test: {X_test.shape}")

    # --- Hyperparameter Optimization (Short Range) ---
    # Takes a params dict and returns a dictionary with 'loss' key (Hyperopt minimizes this)
    def objective(params):
        model = XGBRegressor(**params) # "**" takes each key-value pair in the dictionary and passes it to the XGBoost regressor as if you had typed them manually
        cv = KFold(n_splits=3, shuffle=True, random_state=12345) # Split the training dataset into 3 equal parts (folds). Then train and evaluate the model 3 times
        score = cross_val_score(model, X_train, y_train, cv=cv,# Cross validation and the mean of the MSE
                                scoring='neg_mean_squared_error', n_jobs=-1).mean()
        return {'loss': -score, 'status': STATUS_OK}

    # --- Shorter Ranges for Faster Search ---
    space = {
        'n_estimators': scope.int(hp.quniform('n_estimators', 500, 900, 100)), # quniform (Linear but rounded) creates a uniform distribution
        'max_depth': scope.int(hp.quniform('max_depth', 3, 5, 1)), 
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.05)), # loguniform (Log space): get more density among smaller numbers
        'subsample': hp.uniform('subsample', 0.6, 0.8), # uniform (Linear space)
        'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 0.8),
        'reg_lambda': hp.uniform('reg_lambda', 0.3, 0.5),
        'gamma': hp.uniform('gamma', 0.25, 0.35),
        'random_state': 12345
    }

    print("--- Running Quick Hyperparameter Optimization (10 evals) ---")
    trials = Trials() # Collects results so it can inspect losses / params after the run
    best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10, trials=trials) # Perform 10 trials (small for demo)

    print("Best Parameters Found:")
    print(best_params)

    # --- Train Final Model ---
    print("--- Training Final Model ---")
    final_model = XGBRegressor(
        n_estimators=int(best_params['n_estimators']), # Number of boosting rounds (trees)
        max_depth=int(best_params['max_depth']), # Maximum depth of each tree. Controls model complexity
        learning_rate=best_params['learning_rate'], # Step size shrinkage used in updates for Gradient Descent 
        subsample=best_params['subsample'], # Fraction of training data used per tree. Helps prevent overfitting
        colsample_bytree=best_params['colsample_bytree'], # Fraction of features (columns) sampled per tree. Adds randomness and prevents overfitting
        reg_lambda=best_params['reg_lambda'], # L2 regularization term on weights. Controls model complexity by penalizing large weights
        gamma=best_params['gamma'], # Minimum loss reduction required to make a further partition
        random_state=12345
    )

    final_model.fit(X_train, y_train)

    evaluate_model(final_model, X_train, y_train, "Training")
    evaluate_model(final_model, X_valid, y_valid, "Validation")
    evaluate_model(final_model, X_test, y_test, "Test")

    # --- Save Model ---
    # SageMaker takes everything inside /opt/ml/model after script finishes, tars it into model.tar.gz and uploads to the training job’s ModelArtifacts S3 path automatically
    print("--- Saving Model ---")
    os.makedirs(args.model_dir, exist_ok=True) # args.model_dir maps to SM_MODEL_DIR (e.g. /opt/ml/model inside container)
    model_path = os.path.join(args.model_dir, "model.joblib") # Take the trained model object in memory, compress it, and write it to a binary file named model.joblib
    joblib.dump(final_model, model_path)
    print(f"✅ Model saved to {model_path}")

#The file model.joblib typically includes:
#	•	The XGBoost model weights (learned parameters).
#	•	The hyperparameters (like learning rate, max depth, etc.).
#	•	Metadata for how to reconstruct the model object in memory.

# --- SageMaker Entry Point ---
# It “parses” (reads) the arguments passed by SageMaker into the container’s file
# Then it calls the main() function with those arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Environment variables automatically created by SageMaker inside the container
    # They tell the script (train.py) where to find the data and where to save the model
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN')) # The directory where SageMaker mounted the training data
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR')) # Where your model must be saved (so SageMaker can upload it to S3)

    args = parser.parse_args()
    main(args)