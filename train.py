import mlflow
import os

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Use local MLflow storage
mlflow.set_tracking_uri("file:./mlruns")

# Load data
data = load_iris()
X = data.data
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
with mlflow.start_run() as run:

    model = RandomForestClassifier(n_estimators=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    # Log accuracy to MLflow
    mlflow.log_metric("accuracy", accuracy)

    # Get run ID
    run_id = run.info.run_id

    # Save run ID
    with open("model_info.txt", "w") as f:
        f.write(run_id)

print("Training completed")
