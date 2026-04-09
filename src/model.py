from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

def get_model_pipeline(preprocess):

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", model)
    ])

    return pipeline