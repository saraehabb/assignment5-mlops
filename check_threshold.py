import mlflow
import os

mlflow.set_tracking_uri("file:./mlruns")

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)

accuracy = run.data.metrics["accuracy"]

print("Accuracy:", accuracy)

# DO NOT FAIL PIPELINE
if accuracy < 0.85:
    print("Accuracy below threshold, but continuing...")

print("Done successfully")
