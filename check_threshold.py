import mlflow
import os
import sys

mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)

accuracy = run.data.metrics["accuracy"]

print("Accuracy:", accuracy)

if accuracy < 0.85:
    sys.exit(1)
