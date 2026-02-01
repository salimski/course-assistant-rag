"""
Weather Tool - FREE weather via Open-Meteo (no API key required)
Supports:
- Current weather
- Daily forecast for a specific date (YYYY-MM-DD)
"""

from __future__ import annotations
import requests
from datetime import datetime
from typing import Optional


class WeatherTool:
    def __init__(self):
        self.geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
        self.forecast_url = "https://api.open-meteo.com/v1/forecast"

    def get_weather(self, location: str, on_date: Optional[str] = None) -> str:
        """
        Args:
            location: City name (e.g., "Haifa, Israel")
            on_date: Optional "YYYY-MM-DD". If provided, returns daily forecast for that date.
        """
        # 1) Geocode
        geo = self._geocode(location)
        if isinstance(geo, str):
            return geo

        lat, lon, resolved_name, country = geo

        # 2) Forecast or current
        if on_date:
            return self._daily_forecast(lat, lon, resolved_name, country, on_date)
        return self._current_weather(lat, lon, resolved_name, country)

    def _geocode(self, location: str):
        try:
            r = requests.get(
                self.geocode_url,
                params={"name": location, "count": 1, "language": "en", "format": "json"},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            results = data.get("results", [])
            if not results:
                return f"Error: Location '{location}' not found. Try a more specific name (e.g., 'Haifa, Israel')."

            top = results[0]
            return (
                float(top["latitude"]),
                float(top["longitude"]),
                top.get("name", location),
                top.get("country", ""),
            )
        except requests.exceptions.RequestException as e:
            return f"Error connecting to geocoding service: {e}"
        except (KeyError, ValueError) as e:
            return f"Error parsing geocoding response: {e}"

    def _current_weather(self, lat: float, lon: float, name: str, country: str) -> str:
        try:
            r = requests.get(
                self.forecast_url,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current_weather": "true",
                    "timezone": "auto",
                },
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            cw = data.get("current_weather")
            if not cw:
                return "Error: Weather service returned no current weather."

            temp = cw.get("temperature")
            wind = cw.get("windspeed")
            code = cw.get("weathercode")
            t = cw.get("time")

            return (
                f"Current weather in {name}{', ' + country if country else ''}:\n"
                f"- Temperature: {temp}°C\n"
                f"- Wind: {wind} km/h\n"
                f"- Weather code: {code}\n"
                f"- Time: {t}"
            )
        except requests.exceptions.RequestException as e:
            return f"Error connecting to weather service: {e}"
        except (KeyError, ValueError) as e:
            return f"Error parsing weather data: {e}"

    def _daily_forecast(self, lat: float, lon: float, name: str, country: str, on_date: str) -> str:
        try:
            # validate date format
            datetime.strptime(on_date, "%Y-%m-%d")
        except ValueError:
            return "Error: on_date must be in YYYY-MM-DD format (e.g., 2026-02-10)."

        try:
            r = requests.get(
                self.forecast_url,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,wind_speed_10m_max",
                    "timezone": "auto",
                },
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()

            daily = data.get("daily", {})
            days = daily.get("time", [])
            if not days:
                return "Error: Weather service returned no daily forecast."

            if on_date not in days:
                return (
                    f"Forecast not available for {on_date}.\n"
                    f"Available forecast range: {days[0]} to {days[-1]}.\n"
                    f"Try again closer to the date."
                )

            idx = days.index(on_date)
            tmax = daily.get("temperature_2m_max", [None])[idx]
            tmin = daily.get("temperature_2m_min", [None])[idx]
            pop = daily.get("precipitation_probability_max", [None])[idx]
            wind = daily.get("wind_speed_10m_max", [None])[idx]

            return (
                f"Forecast for {name}{', ' + country if country else ''} on {on_date}:\n"
                f"- Temperature: {tmin}°C to {tmax}°C\n"
                f"- Max precipitation probability: {pop}%\n"
                f"- Max wind: {wind} km/h"
            )
        except requests.exceptions.RequestException as e:
            return f"Error connecting to weather service: {e}"
        except (KeyError, ValueError, IndexError) as e:
            return f"Error parsing forecast data: {e}"


WEATHER_TOOL_DESCRIPTION = """
Get weather for a city.
- If user asks current conditions: call with location only.
- If user asks about a specific day (e.g., exam day): pass on_date in YYYY-MM-DD.
"""


def create_weather_tool():
    return WeatherTool()
