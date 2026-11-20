import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple


_PATTERNS = [
    # Android 24h / 12h: "15/11/26, 09:12 - User: msg" or "15/11/2026, 9:12 pm - User: msg"
    re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}(?:\s*[apAP][mM])?)\s*-\s*(.*)$"),
    # iOS bracket format: "[15/11/26, 09:12:05] User: msg"
    re.compile(r"^\[(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:\s*[apAP][mM])?)\]\s*(.*)$"),
]

_DATE_FORMATS = (
    "%d/%m/%y, %H:%M",
    "%d/%m/%Y, %H:%M",
    "%d/%m/%y, %I:%M %p",
    "%d/%m/%Y, %I:%M %p",
    "%d/%m/%y, %H:%M:%S",
    "%d/%m/%Y, %H:%M:%S",
    "%m/%d/%y, %I:%M %p",
    "%m/%d/%Y, %I:%M %p",
)


def _parse_timestamp(raw: str) -> Optional[datetime]:
    cleaned = re.sub(r"\s+", " ", raw.strip()).upper()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def parse_chat_records(data: str) -> List[Dict[str, object]]:
    """Parse raw WhatsApp export text into structured message dicts (no pandas required)."""
    entries: List[Tuple[datetime, str]] = []
    for raw_line in data.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        matched = False
        for pattern in _PATTERNS:
            m = pattern.match(line)
            if m:
                ts = _parse_timestamp(m.group(1))
                if ts is not None:
                    entries.append((ts, m.group(2)))
                    matched = True
                    break
        if not matched and entries:
            # Multi-line continuation of previous message
            prev_ts, prev_body = entries[-1]
            entries[-1] = (prev_ts, f"{prev_body}\n{line}")

    records: List[Dict[str, object]] = []
    for ts, body in entries:
        if ": " in body:
            user, msg = body.split(": ", 1)
        else:
            user, msg = "group_notification", body

        hour = ts.hour
        period = f"{hour:02d}-{(hour + 1) % 24:02d}"
        records.append({
            "date": ts,
            "user": user.strip(),
            "message": msg.strip(),
            "only_date": ts.date(),
            "year": ts.year,
            "month_num": ts.month,
            "month": ts.strftime("%B"),
            "day": ts.day,
            "day_name": ts.strftime("%A"),
            "hour": hour,
            "minute": ts.minute,
            "period": period,
        })
    return records


def preprocess(data: str):
    """Return a pandas DataFrame of parsed WhatsApp messages."""
    import pandas as pd

    records = parse_chat_records(data)
    if not records:
        return pd.DataFrame(
            columns=["date", "user", "message", "only_date", "year", "month_num", "month", "day", "day_name", "hour", "minute", "period"]
        )
    return pd.DataFrame(records)
