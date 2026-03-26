import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Use local MLflow storage
mlflow.set_tracking_uri("file:./mlruns")

# Load dataset
data = load_iris()
X = data.data
y = data.target

# make training deterministic
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Start MLflow run
with mlflow.start_run() as run:

    # deterministic model
    model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    random_state=42
)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Log to MLflow
    mlflow.log_metric("accuracy", accuracy)

    # Get run ID
    run_id = run.info.run_id

    # Save run ID (IMPORTANT for pipeline)
    with open("model_info.txt", "w") as f:
        f.write(run_id)

print("Training completed successfully")
