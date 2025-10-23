import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import mlflow
import joblib

# Set MLflow experiment name
mlflow.set_experiment("Titanic Survival Prediction")

# Load preprocessed data
df = pd.read_csv('data/train.csv')

# Separate features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
with mlflow.start_run(run_name="Logistic Regression Baseline"):
    # Set tag
    mlflow.set_tag("model_type", "Logistic Regression")
    
    # Define parameters
    params = {"solver": "liblinear", "random_state": 42}
    mlflow.log_params(params)
    
    # Train model
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)
    
    # Make predictions
    y_pred = lr.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("auc", auc)
    
    # Save and log model
    joblib.dump(lr, 'model.joblib')
    mlflow.log_artifact('model.joblib')
    
    print(f"Training complete! Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
