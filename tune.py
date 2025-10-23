import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import mlflow

# Set MLflow experiment
mlflow.set_experiment("Titanic Survival Prediction")

# Load preprocessed data
df = pd.read_csv('data/train.csv')
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameters to test
n_estimators_options = [50, 100, 200]
max_depth_options = [5, 10, 15]

# Parent run for tuning
with mlflow.start_run(run_name="RandomForest Tuning"):
    mlflow.set_tag("model_type", "RandomForest Hyperparameter Tuning")
    
    best_accuracy = 0
    best_params = {}
    
    # Nested runs for each hyperparameter combination
    for n_est in n_estimators_options:
        for max_d in max_depth_options:
            with mlflow.start_run(run_name=f"RF_n{n_est}_d{max_d}", nested=True):
                # Log parameters
                mlflow.log_param("n_estimators", n_est)
                mlflow.log_param("max_depth", max_d)
                
                # Train model
                rf = RandomForestClassifier(n_estimators=n_est, max_depth=max_d, random_state=42)
                rf.fit(X_train, y_train)
                
                # Make predictions
                y_pred = rf.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
                
                # Log metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("auc", auc)
                
                print(f"n_estimators={n_est}, max_depth={max_d}: Accuracy={accuracy:.4f}, AUC={auc:.4f}")
                
                # Track best model
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {"n_estimators": n_est, "max_depth": max_d}
    
    # Log best parameters in parent run
    mlflow.log_params(best_params)
    mlflow.log_metric("best_accuracy", best_accuracy)
    
    print(f"\nBest params: {best_params} with accuracy: {best_accuracy:.4f}")
