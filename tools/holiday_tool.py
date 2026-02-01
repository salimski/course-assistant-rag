"""
Holiday Tool - Public holidays via Hebcal API (FREE, no API key)

Good for Israel:
https://www.hebcal.com/hebcal?cfg=json&v=1&year=YYYY&maj=on&geo=country&country=IL
"""

from __future__ import annotations

import requests
from datetime import datetime, date
from typing import Optional, List, Dict


class HolidayTool:
    def __init__(self, default_country_code: str = "IL"):
        self.default_country_code = default_country_code.upper().strip()
        self.base_url = "https://www.hebcal.com/hebcal"
        self.headers = {
            "Accept": "application/json",
            "User-Agent": "course-assistant-rag/1.0 (student project)"
        }

    def _fetch_year(self, year: int, country_code: str) -> (Optional[List[Dict]], Optional[str]):
        cc = country_code.upper().strip()

        # For the assignment, IL is enough + stable.
        if cc != "IL":
            return None, "Hebcal tool currently supports only country_code='IL'."

        params = {
            "cfg": "json",
            "v": "1",
            "year": str(year),
            "maj": "on",   # major holidays
            "min": "off",
            "mod": "off",
            "nx": "off",
            "mf": "off",
            "ss": "off",
            "c": "on",
            "geo": "country",
            "country": "IL",
        }

        try:
            r = requests.get(self.base_url, params=params, headers=self.headers, timeout=15)
            if r.status_code != 200:
                snippet = (r.text or "")[:200].replace("\n", " ")
                return None, f"Holiday API error: HTTP {r.status_code}. Response: {snippet}"

            if not r.text or not r.text.strip():
                return None, "Holiday API error: Empty response (no JSON returned)."

            try:
                data = r.json()
            except Exception as e:
                snippet = r.text[:200].replace("\n", " ")
                return None, f"Holiday API error: Failed to parse JSON ({e}). Response starts with: {snippet}"

            items = data.get("items", [])
            if not isinstance(items, list):
                return None, "Holiday API error: JSON missing 'items' list."

            return items, None

        except requests.exceptions.RequestException as e:
            return None, f"Error connecting to holidays service: {e}"

    def get_holidays(self, year: int, country_code: str = "IL") -> str:
        print(f"\nFetching public holidays for: {country_code}, {year}")
        items, err = self._fetch_year(year, country_code)
        if err:
            return err

        holidays = [it for it in items if it.get("category") == "holiday"]
        if not holidays:
            return f"No holidays found for {country_code} in {year}."

        lines = [f"Holidays in {country_code} for {year} (Hebcal):"]
        for it in holidays[:60]:
            ds = it.get("date", "")
            title = it.get("title", "")
            lines.append(f"- {ds}: {title}")

        if len(holidays) > 60:
            lines.append(f"...and {len(holidays) - 60} more.")

        return "\n".join(lines)

    def is_holiday(self, date_str: str, country_code: str = "IL") -> str:
        try:
            dt = datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
        except ValueError:
            return "Error: date must be YYYY-MM-DD (e.g., 2026-02-10)."

        print(f"\nChecking if {date_str} is a holiday in {country_code}...")
        items, err = self._fetch_year(dt.year, country_code)
        if err:
            return err

        for it in items:
            if it.get("category") != "holiday":
                continue
            if it.get("date") == date_str:
                return f"Yes — {date_str} is a holiday in {country_code}: {it.get('title','')}."

        return f"No — {date_str} is not a (major) holiday in {country_code}."

    def get_next_holiday(self, country_code: str = "IL", today_iso: Optional[str] = None) -> str:
        if today_iso:
            try:
                today = datetime.strptime(today_iso.strip(), "%Y-%m-%d").date()
            except ValueError:
                return "Error: today_iso must be YYYY-MM-DD."
        else:
            today = date.today()

        print(f"\nFinding next holiday in {country_code} after {today.isoformat()}...")

        items, err = self._fetch_year(today.year, country_code)
        if err:
            return err

        candidates = []
        for it in items:
            if it.get("category") != "holiday":
                continue
            ds = it.get("date", "")
            try:
                d = datetime.strptime(ds, "%Y-%m-%d").date()
            except Exception:
                continue
            if d > today:
                candidates.append((d, it.get("title", "")))

        if not candidates:
            items2, err2 = self._fetch_year(today.year + 1, country_code)
            if err2:
                return f"No more holidays in {today.year}. Also failed fetching {today.year+1}: {err2}"

            for it in items2:
                if it.get("category") != "holiday":
                    continue
                ds = it.get("date", "")
                try:
                    d = datetime.strptime(ds, "%Y-%m-%d").date()
                except Exception:
                    continue
                candidates.append((d, it.get("title", "")))

        if not candidates:
            return "No upcoming holidays found."

        candidates.sort(key=lambda x: x[0])
        d, title = candidates[0]
        return f"Next holiday in {country_code}: {d.isoformat()} — {title}."


def create_holiday_tool():
    return HolidayTool()
