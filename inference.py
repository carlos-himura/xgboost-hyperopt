import os
import boto3
import pandas as pd
from sagemaker import Session
from sagemaker.model import Model
from preprocessing import preprocess_data

def run_batch_inference(model_artifact, container, local_data_path, bucket, prefix):
    session = Session()
    role = session.get_caller_identity_arn()
    import sagemaker
    role = sagemaker.get_execution_role()
    
    # Load new (blind) data
    df = pd.read_csv(local_data_path)
    print(f"Loaded data shape: {df.shape}")

    # Apply preprocessing (no target)
    df_processed = preprocess_data(df, is_training=False)
    
    # Save locally for transform
    local_batch_file = "batch_data_for_transform.csv"
    df_processed.to_csv(local_batch_file, header=False, index=False)
    print(f"Saved preprocessed file: {local_batch_file}")

    # Upload to S3
    s3 = boto3.Session().resource("s3")
    s3_path = os.path.join(prefix, local_batch_file)
    s3.Bucket(bucket).Object(s3_path).upload_file(local_batch_file)
    s3_uri = f"s3://{bucket}/{s3_path}"
    print(f"✅ Uploaded to {s3_uri}")

    # --- Create Model object from previous training job ---
    model = Model(
    image_uri=container,           # Same container used during training
    model_data=model_artifact,     # Path to model.tar.gz from your previous job
    role=role,
    sagemaker_session=session
    )

    # Create transformer
    transformer = model.transformer(
        instance_count=1,
        instance_type="ml.m4.xlarge",
        strategy="SingleRecord",
        assemble_with="Line",
        output_path=f"s3://{bucket}/{prefix}/predictions"
    )

    # 6️⃣ Run batch transform
    print("Starting Batch Transform job...")
    transformer.transform(
        data=s3_uri,
        content_type="text/csv",
        split_type="Line"
    )
    transformer.wait()
    print("✅ Batch Transform complete.")

    # 7️⃣ Download predictions
    output_path = f"{prefix}/predictions"
    print(f"Predictions stored at: s3://{bucket}/{output_path}")

    return f"s3://{bucket}/{output_path}"