from __future__ import annotations

import os
import requests
from typing import Optional

# OpenWeatherMap credentials
# # set via terminal →  export OPENWEATHER_API_KEY="your_key"
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

# Endpoint exactly as shown in your confirmation email
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"


def get_weather_severity(city: str = "Boston", api_key: Optional[str] = None) -> float:
    """
    Fetch live weather for `city` and return a severity score in [0.0, 1.0].

    Score mapping
    -------------
    0.0  clear / light clouds
    0.2  drizzle / mist
    0.3  fog / haze / smoke
    0.5  moderate rain
    0.7  heavy snow
    0.9  thunderstorm

    Falls back to 0.0 if the key is not yet activated or the request fails.
    (OpenWeatherMap says activation takes a couple of hours after signup.)
    """
    key = api_key or OPENWEATHER_API_KEY
    if not key:
        print("[WeatherAPI] No API key found. Using severity=0.0.")
        return 0.0

    try:
        resp = requests.get(
            BASE_URL,
            params={"q": city, "appid": key, "units": "metric"},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()

        wid    = data["weather"][0]["id"]
        rain1h = data.get("rain", {}).get("1h", 0.0)
        snow1h = data.get("snow", {}).get("1h", 0.0)

        if   wid >= 800: return 0.0                               # clear / clouds
        elif wid >= 700: return 0.3                               # fog / haze
        elif wid >= 600: return min(1.0, 0.5 + snow1h / 10.0)   # snow
        elif wid >= 500: return min(1.0, 0.3 + rain1h / 10.0)   # rain
        elif wid >= 300: return 0.2                               # drizzle
        elif wid >= 200: return 0.9                               # thunderstorm
        return 0.0

    except requests.exceptions.HTTPError as e:
        # 401 = key not yet activated (can take a few hours after signup)
        if resp.status_code == 401:
            print("[WeatherAPI] Key not yet activated — wait a couple of hours. Using 0.0.")
        else:
            print(f"[WeatherAPI] HTTP error {resp.status_code}: {e}. Using 0.0.")
        return 0.0

    except requests.exceptions.RequestException as e:
        print(f"[WeatherAPI] Request failed: {e}. Using 0.0.")
        return 0.0

    except (KeyError, ValueError) as e:
        print(f"[WeatherAPI] Parse error: {e}. Using 0.0.")
        return 0.0


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    severity = get_weather_severity(city="Boston")
    print(f"Weather severity for Boston: {severity}")

    # You can also test with any other city, e.g.:
    # get_weather_severity(city="London")
    # get_weather_severity(city="New York")