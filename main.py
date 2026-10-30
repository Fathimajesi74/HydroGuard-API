
from pathlib import Path
from datetime import timedelta, datetime, timezone

import joblib
from fastapi import FastAPI, HTTPException

from ml_model.firebase_data import load_sensor_history, SENSOR_FIELDS

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "random_forest_30min.joblib"

STALE_AFTER_SECONDS = 180

app = FastAPI(title="HydroGuard 30-Minute Forecast API")


def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError("30-minute model file not found.")

    saved = joblib.load(MODEL_PATH)

    if saved["sensor_fields"] != SENSOR_FIELDS:
        raise RuntimeError("Sensor field configuration mismatch.")

    model = saved["model"]
    lookback = saved["lookback"]
    forecast_minutes = saved["forecast_minutes"]

    expected_features = lookback * len(SENSOR_FIELDS)

    if saved["feature_count"] != expected_features:
        raise RuntimeError("Invalid model feature count.")

    return model, lookback, forecast_minutes


@app.get("/")
def home():
    return {
        "message": "HydroGuard Forecast API is running",
        "endpoint": "/predict"
    }


@app.get("/health")
def health():
    try:
        model, lookback, forecast_minutes = load_model()
        return {
            "status": "running",
            "model_loaded": True,
            "lookback": lookback,
            "forecast_minutes": forecast_minutes
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.get("/predict")
def predict():
    try:
        model, lookback, forecast_minutes = load_model()
        history = load_sensor_history()

        if len(history) < lookback:
            raise RuntimeError("Not enough sensor readings.")

        latest_time = history.index[-1]
        now = datetime.now(timezone.utc)

        age_seconds = (
            now - latest_time.to_pydatetime()
        ).total_seconds()

        recent_values = history[SENSOR_FIELDS].tail(
            lookback
        ).to_numpy(dtype=float)

        features = recent_values.reshape(1, -1)
        predicted_values = model.predict(features)[0]

        prediction = {
            field: round(float(value), 2)
            for field, value in zip(
                SENSOR_FIELDS, predicted_values
            )
        }

        prediction["ph"] = max(0, min(14, prediction["ph"]))
        prediction["waterLevel"] = max(
            0, min(100, prediction["waterLevel"])
        )

        current = {
            field: round(float(value), 2)
            for field, value in zip(
                SENSOR_FIELDS, recent_values[-1]
            )
        }

        issues = []

        if prediction["temperature"] < 18 or prediction["temperature"] > 30:
            issues.append("Temperature outside selected range")

        if prediction["ph"] < 5.5:
            issues.append("pH is low")
        elif prediction["ph"] > 6.5:
            issues.append("pH is high")

        if prediction["waterLevel"] < 50:
            issues.append("Water level is low")

        predicted_for = latest_time + timedelta(
            minutes=forecast_minutes
        )

        return {
            "success": True,
            "forecast": f"{forecast_minutes} minutes",
            "data_status": (
                "STALE_OFFLINE"
                if age_seconds > STALE_AFTER_SECONDS
                else "LIVE"
            ),
            "is_stale": age_seconds > STALE_AFTER_SECONDS,
            "data_age_seconds": round(max(0, age_seconds), 1),
            "based_on": latest_time.isoformat(),
            "predicted_for": predicted_for.isoformat(),
            "current": current,
            "prediction": prediction,
            "condition": "Needs Attention" if issues else "Suitable",
            "issues": issues
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
