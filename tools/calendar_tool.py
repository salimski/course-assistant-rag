"""
Calendar Tool - JSON-backed calendar events (persistent across runs)

- Loads events from a JSON file on startup
- Creates the JSON file with default events if missing
- Supports add/remove/update with automatic saving

This remains a mock calendar (not Google Calendar), but now it is persistent.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional


class CalendarTool:
    def __init__(self, json_path: Optional[str] = None):
        """
        Args:
            json_path: optional path to the JSON file.
                      If None, defaults to <project_root>/data/calendar_events.json
        """
        project_root = Path(__file__).resolve().parents[1]  # tools/ -> project root
        default_path = project_root / "data" / "calendar_events.json"
        self.json_path = Path(json_path) if json_path else default_path

        # Ensure data directory exists
        self.json_path.parent.mkdir(parents=True, exist_ok=True)

        # Load (or create) events
        self.events: List[Dict] = []
        self._load_or_init()

    # -------------------------
    # Persistence
    # -------------------------
    def _default_events(self) -> List[Dict]:
        return [
            {
                "title": "Information Retrieval Final Project",
                "date": "2026-02-15",
                "time": "23:59",
                "type": "deadline"
            },
            {
                "title": "IR Lecture - Advanced Ranking",
                "date": "2026-02-03",
                "time": "10:00",
                "type": "class"
            },
            {
                "title": "Machine Learning Exam",
                "date": "2026-02-10",
                "time": "09:00",
                "type": "exam"
            },
            {
                "title": "Study Group - RAG Systems",
                "date": "2026-02-05",
                "time": "14:00",
                "type": "meeting"
            }
        ]

    def _load_or_init(self) -> None:
        if self.json_path.exists():
            self.events = self._load_events()
        else:
            # First run: create with defaults
            self.events = self._default_events()
            self._save_events()

    def _load_events(self) -> List[Dict]:
        try:
            with self.json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("calendar_events.json must contain a list of events")
            # Basic validation/filter
            cleaned = []
            for e in data:
                if self._is_valid_event(e):
                    cleaned.append(e)
            return cleaned
        except Exception as e:
            # If file is corrupted, fall back safely (do not crash the whole app)
            print(f"⚠ Warning: Failed to load calendar JSON ({self.json_path}). Reason: {e}")
            print("⚠ Falling back to default events (and rewriting the JSON).")
            events = self._default_events()
            self.events = events
            self._save_events()
            return events

    def _save_events(self) -> None:
        try:
            with self.json_path.open("w", encoding="utf-8") as f:
                json.dump(self.events, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠ Warning: Failed to save calendar JSON ({self.json_path}). Reason: {e}")

    # -------------------------
    # Validation helpers
    # -------------------------
    def _is_valid_event(self, e: Dict) -> bool:
        if not isinstance(e, dict):
            return False
        required = {"title", "date", "time", "type"}
        if not required.issubset(set(e.keys())):
            return False
        if not isinstance(e["title"], str) or not e["title"].strip():
            return False
        if not isinstance(e["type"], str) or not e["type"].strip():
            return False
        # Validate date/time formats
        try:
            datetime.strptime(e["date"], "%Y-%m-%d")
            datetime.strptime(e["time"], "%H:%M")
        except Exception:
            return False
        return True

    # -------------------------
    # CRUD (for your future tools)
    # -------------------------
    def add_event(self, title: str, date_str: str, time_str: str, event_type: str) -> str:
        """
        Add an event and persist to JSON.
        """
        event = {
            "title": title.strip(),
            "date": date_str.strip(),
            "time": time_str.strip(),
            "type": event_type.strip().lower()
        }

        if not self._is_valid_event(event):
            return "Error: Invalid event. Use date YYYY-MM-DD and time HH:MM."

        # Avoid exact duplicates
        for e in self.events:
            if (e["title"].lower() == event["title"].lower()
                and e["date"] == event["date"]
                and e["time"] == event["time"]
                and e["type"] == event["type"]):
                return "Event already exists."

        self.events.append(event)
        self.events.sort(key=lambda x: (x["date"], x["time"]))
        self._save_events()
        return f"Added: {event['title']} on {event['date']} at {event['time']} ({event['type']})."

    def remove_event(self, title: str) -> str:
        """
        Remove events whose title matches (case-insensitive). Persists changes.
        """
        t = title.strip().lower()
        if not t:
            return "Error: Please provide a title to remove."

        before = len(self.events)
        self.events = [e for e in self.events if e["title"].strip().lower() != t]
        removed = before - len(self.events)

        if removed == 0:
            return f"No event found with title '{title}'."

        self._save_events()
        return f"Removed {removed} event(s) titled '{title}'."

    # -------------------------
    # Existing methods (used by your agent)
    # -------------------------
    def get_upcoming_events(self, days: int = 7) -> str:
        print(f"\nChecking calendar for next {days} days...")

        today = datetime.now()
        cutoff_date = today + timedelta(days=days)

        upcoming = []
        for event in self.events:
            event_date = datetime.strptime(event["date"], "%Y-%m-%d")
            if today <= event_date <= cutoff_date:
                upcoming.append(event)

        if not upcoming:
            return f"No events scheduled in the next {days} days."

        upcoming.sort(key=lambda x: (x["date"], x["time"]))

        result = f"Upcoming events (next {days} days):\n\n"
        for event in upcoming:
            event_date = datetime.strptime(event["date"], "%Y-%m-%d")
            days_until = (event_date - today).days

            if days_until == 0:
                time_str = "Today"
            elif days_until == 1:
                time_str = "Tomorrow"
            else:
                time_str = f"In {days_until} days"

            result += f"📅 {event['title']}\n"
            result += f"   {time_str} ({event['date']}) at {event['time']}\n"
            result += f"   Type: {event['type'].capitalize()}\n\n"

        return result.strip()

    def check_specific_event(self, query: str) -> str:
        print(f"\nSearching calendar for: '{query}'")

        query_lower = query.lower().strip()
        matches = []

        for event in self.events:
            if (query_lower in event["title"].lower() or
                query_lower in event["type"].lower()):
                matches.append(event)

        if not matches:
            return f"No events found matching '{query}'."

        matches.sort(key=lambda x: (x["date"], x["time"]))

        result = f"Found {len(matches)} event(s) matching '{query}':\n\n"
        for event in matches:
            result += f"📅 {event['title']}\n"
            result += f"   Date: {event['date']} at {event['time']}\n"
            result += f"   Type: {event['type'].capitalize()}\n\n"

        return result.strip()

    def get_next_deadline(self) -> str:
        print("\nChecking for next deadline...")

        today = datetime.now()
        deadlines = [e for e in self.events if e.get("type") == "deadline"]

        future_deadlines = []
        for d in deadlines:
            deadline_date = datetime.strptime(d["date"], "%Y-%m-%d")
            if deadline_date >= today:
                future_deadlines.append(d)

        if not future_deadlines:
            return "No upcoming deadlines found."

        future_deadlines.sort(key=lambda x: (x["date"], x["time"]))
        next_deadline = future_deadlines[0]

        deadline_date = datetime.strptime(next_deadline["date"], "%Y-%m-%d")
        days_until = (deadline_date - today).days

        if days_until == 0:
            time_str = "Today!"
        elif days_until == 1:
            time_str = "Tomorrow!"
        else:
            time_str = f"In {days_until} days"

        result = "⚠️  Next Deadline:\n"
        result += f"{next_deadline['title']}\n"
        result += f"Due: {next_deadline['date']} at {next_deadline['time']}\n"
        result += f"Time remaining: {time_str}"
        return result

    def get_next_exam_json(self) -> str:
        """
        Returns JSON for the next upcoming exam (used by your agent chaining).
        """
        now = datetime.now()
        exams = []
        for e in self.events:
            if e.get("type") != "exam":
                continue
            d = datetime.strptime(e["date"], "%Y-%m-%d")
            if d >= now:
                exams.append(e)

        exams.sort(key=lambda x: (x["date"], x["time"]))
        if not exams:
            return json.dumps({"found": False, "reason": "No upcoming exams found."})

        e = exams[0]
        return json.dumps({
            "found": True,
            "title": e["title"],
            "date": e["date"],
            "time": e["time"],
            "type": e["type"]
        })


def create_calendar_tool():
    return CalendarTool()
