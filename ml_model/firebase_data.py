
import requests
import pandas as pd

FIREBASE_URL = (
    "https://hydroguard-3b02f-default-rtdb."
    "asia-southeast1.firebasedatabase.app"
)

SENSOR_FIELDS = [
    "temperature",
    "turbidity",
    "ph",
    "waterLevel"
]


def normalise_record(record):
    if not isinstance(record, dict):
        return None

    aliases = {
        "temperature": ["temperature", "temperature_C"],
        "turbidity": ["turbidity", "turbidity_reading"],
        "ph": ["ph", "pH"],
        "waterLevel": ["waterLevel", "water_level_percent"]
    }

    result = {}

    for field, names in aliases.items():
        value = next(
            (record[name] for name in names if name in record),
            None
        )

        try:
            result[field] = float(value)
        except (TypeError, ValueError):
            return None

    try:
        timestamp = float(record["timestamp"])
    except (KeyError, TypeError, ValueError):
        return None

    if timestamp > 100_000_000_000:
        timestamp /= 1000

    result["timestamp"] = pd.to_datetime(
        timestamp,
        unit="s",
        utc=True
    )

    return result


def load_sensor_history():

    url = f"{FIREBASE_URL}/sensorHistory.json"

    response = requests.get(url, timeout=20)
    response.raise_for_status()

    data = response.json()

    if not data:
        raise RuntimeError("No sensor history found in Firebase.")

    records = (
        list(data.values())
        if isinstance(data, dict)
        else data
    )

    rows = [
        normalise_record(record)
        for record in records
    ]

    rows = [row for row in rows if row is not None]

    if not rows:
        raise RuntimeError("No valid sensor records found.")

    df = pd.DataFrame(rows)

    df = df.sort_values("timestamp")
    df = df.drop_duplicates("timestamp", keep="last")
    df = df.set_index("timestamp")

    now = pd.Timestamp.now(tz="UTC")

    df = df[
        (df.index >= now - pd.Timedelta(days=7)) &
        (df.index <= now + pd.Timedelta(minutes=5))
    ]

    if df.empty:
        raise RuntimeError("No recent valid Firebase readings found.")

    df = df[SENSOR_FIELDS].dropna()

    print("Valid recent Firebase readings:", len(df))
    print("Oldest reading:", df.index.min())
    print("Latest reading:", df.index.max())

    return df
