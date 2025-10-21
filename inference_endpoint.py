import os
import boto3
import pandas as pd
from sagemaker import Session
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import CSVDeserializer
from preprocessing import preprocess_data
from datetime import datetime
import sagemaker

def run_endpoint_inference(
    model_artifact,
    container,
    local_data_path,
    bucket,
    endpoint_name=None,
    deploy_if_missing=True
):
    """
    Run inference on a SageMaker endpoint.
    
    Parameters:
    - model_artifact: S3 path to model.tar.gz
    - container: Docker image URI
    - local_data_path: CSV with input data
    - bucket: S3 bucket for saving predictions
    - endpoint_name: optional custom endpoint name
    - deploy_if_missing: deploy model if endpoint doesn't exist
    """

    session = Session()

    # --- Get execution role ---
    try:
        role = sagemaker.get_execution_role()
    except Exception:
        role = os.environ.get(
            "SAGEMAKER_ROLE",
            "arn:aws:iam::<your-account-id>:role/<your-sagemaker-role>"
        )

    # --- Load and preprocess data ---
    df = pd.read_csv(local_data_path)
    print(f"Loaded data shape: {df.shape}")

    df_processed = preprocess_data(df, is_training=False)
    local_inference_file = "inference_data.csv"
    df_processed.to_csv(local_inference_file, header=False, index=False)
    print(f"Saved preprocessed file: {local_inference_file}")

    # --- Prepare predictor ---
    predictor = None
    if endpoint_name:
        try:
            predictor = Predictor(
                endpoint_name=endpoint_name,
                serializer=CSVSerializer(),
                deserializer=CSVDeserializer()
            )
            print(f"✅ Using existing endpoint: {endpoint_name}")
        except Exception:
            predictor = None
            print(f"⚠️ Could not find endpoint: {endpoint_name}")
    else:
        # Generate a default endpoint name
        endpoint_name = f"xgb-endpoint-demo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # --- Deploy model if missing ---
    if predictor is None and deploy_if_missing:
        print(f"🚀 Deploying model to endpoint: {endpoint_name} ...")
        model = Model(
            image_uri=container,
            model_data=model_artifact,
            role=role,
            sagemaker_session=session
        )
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type="ml.m4.xlarge",
            endpoint_name=endpoint_name,
            wait=False  # We'll wait manually
        )

        # Wait for endpoint to be InService
        print("⏳ Waiting for endpoint to be InService ...")
        session.sagemaker_client.get_waiter('endpoint_in_service').wait(EndpointName=endpoint_name)
        print("✅ Endpoint is now InService")

        predictor.serializer = CSVSerializer()
        predictor.deserializer = CSVDeserializer()
        print(f"✅ Model deployed. Predictor ready: {predictor.endpoint_name}")

    if predictor is None:
        raise ValueError("❌ Predictor is None. Cannot continue inference.")

    # --- Run inference ---
    with open(local_inference_file, "r") as f:
        payload = f.read().strip()

    print(f"Payload sample: {payload[:200]}...")
    result = predictor.predict(payload)

    if isinstance(result, bytes):
        result = result.decode("utf-8")

    predictions = [float(x) for x in result.strip().split("\n") if x]
    df_predictions = pd.DataFrame(predictions, columns=["Prediction"])
    print(df_predictions.head())

    # --- Save predictions locally ---
    output_name = f"predictions_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    df_predictions.to_csv(output_name, index=False)
    print(f"Predictions saved locally: {output_name}")

    # --- Upload predictions to S3 ---
    s3 = boto3.client("s3")
    s3_prefix = "xgboost-output/predictions"
    s3_key = f"{s3_prefix}/{output_name}"
    s3.upload_file(output_name, bucket, s3_key)
    s3_uri = f"s3://{bucket}/{s3_key}"
    print(f"✅ Predictions uploaded to: {s3_uri}")

    return df_predictions, predictor