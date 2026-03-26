# FIRST try MLflow
import mlflow

try:
    mlflow.set_tracking_uri("file:./mlruns")

    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    accuracy = run.data.metrics["accuracy"]

    print("Accuracy from MLflow:", accuracy)

except:
    print("MLflow failed → fallback to file")

    with open("accuracy.txt", "r") as f:
        accuracy = float(f.read())

    print("Accuracy from file:", accuracy)

# FINAL DECISION
if accuracy < 0.85:
    print("Accuracy below threshold → FAIL")
    exit(1)

print("Accuracy above threshold → PASS")
