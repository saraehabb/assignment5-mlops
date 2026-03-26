import mlflow

# point to downloaded mlruns
mlflow.set_tracking_uri("file:./mlruns")

# Read run_id
with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)

accuracy = run.data.metrics["accuracy"]

print("Accuracy:", accuracy)

if accuracy < 0.85:
    print("Accuracy below threshold → FAIL")
    exit(1)

print("Accuracy above threshold → PASS")
