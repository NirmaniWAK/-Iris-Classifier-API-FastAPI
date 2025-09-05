import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import json

def train_and_save_model():
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = iris.target_names.tolist()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pipeline (scaler + logistic regression)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200))
    ])

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy:.2f}")

    # Save model and metadata
    joblib.dump({
        "model": model,
        "class_names": class_names,
        "accuracy": accuracy
    }, "model.pkl")
    print("✅ Model saved to model.pkl")

if __name__ == "__main__":
    train_and_save_model()
