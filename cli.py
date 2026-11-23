"""Command-line chat sentiment & engagement summary runner."""

import argparse
import json
from pathlib import Path

from preprocessor import parse_chat_records
from sentiment import analyze_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze WhatsApp chat export sentiment & stats.")
    parser.add_argument("file", help="Path to exported WhatsApp .txt file")
    args = parser.parse_args()

    text = Path(args.file).read_text(encoding="utf-8", errors="ignore")
    records = parse_chat_records(text)
    report = analyze_records(records)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
