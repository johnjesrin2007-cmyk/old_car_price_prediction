import os
import mlflow
import mlflow.sklearn
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_model(pipeline, X, y):
    
    # -------------------------
    # 🔥 MLFLOW SETUP
    # -------------------------
    mlflow.set_experiment("student_marks_prediction")

    # -------------------------
    # 🔀 TRAIN TEST SPLIT
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------
    # 🔁 CROSS VALIDATION (BASE CHECK)
    # -------------------------
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=kfold, scoring="r2"
    )

    print("CV Scores:", cv_scores)
    print("Average CV R2:", np.mean(cv_scores))

    # -------------------------
    # 🎯 HYPERPARAMETER TUNING
    # -------------------------
    param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

    grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1
)
    # -------------------------
    # 🚀 MLFLOW RUN
    # -------------------------
    with mlflow.start_run():

        # Train with tuning
        random_search.fit(X_train, y_train)

        # Get best model
        best_model = random_search.best_estimator_

        # -------------------------
        # 🔮 PREDICT
        # -------------------------
        y_pred = best_model.predict(X_test)

        # -------------------------
        # 📊 METRICS
        # -------------------------
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # -------------------------
        # 🧾 LOG PARAMETERS
        # -------------------------
        mlflow.log_param("model", "RandomForestRegressor")
        mlflow.log_param("cv_folds", 5)
        mlflow.log_params(random_search.best_params_)
        mlflow.log_param("num_rows", X.shape[0])
        mlflow.log_param("num_features", X.shape[1])

        # -------------------------
        # 📊 LOG METRICS
        # -------------------------
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("CV_R2_mean", np.mean(cv_scores))
        

        print("\n------ MODEL METRICS ------")
        print("MAE :", mae)
        print("MSE :", mse)
        print("RMSE:", rmse)
        print("R2 Score:", r2)

        # -------------------------
        # 💾 LOG MODEL (MLFLOW)
        # -------------------------
        mlflow.sklearn.log_model(best_model, "model")

    # -------------------------
    # 💾 SAVE MODEL LOCALLY
    # -------------------------
    os.makedirs("model", exist_ok=True)
    joblib.dump(best_model, "model/model.pkl")

    print("\n✅ Best pipeline saved at model/model.pkl")

    return best_model