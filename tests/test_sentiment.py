from pathlib import Path
import unittest

from preprocessor import parse_chat_records
from sentiment import analyze_records, score_message


class TestChatSentimentEngine(unittest.TestCase):
    def test_positive_negative_and_negation(self):
        self.assertEqual(score_message("Great work, deploy was super smooth!").label, "positive")
        self.assertEqual(score_message("Frustrated with this broken timeout bug").label, "negative")
        self.assertEqual(score_message("not good at all").label, "negative")
        self.assertTrue(score_message("you are an idiot").is_toxic)

    def test_hinglish_sentiment(self):
        self.assertEqual(score_message("bahut badhiya kaam kiya").label, "positive")
        self.assertEqual(score_message("ye bilkul bakwas hai").label, "negative")

    def test_parse_multi_format_and_sample_chat(self):
        raw = Path("sample_chat.txt").read_text(encoding="utf-8")
        records = parse_chat_records(raw)
        self.assertEqual(len(records), 8)

        report = analyze_records(records)
        self.assertEqual(report["total_messages"], 8)
        self.assertGreater(report["distribution"]["positive"], 0)
        self.assertGreater(report["distribution"]["negative"], 0)
        self.assertIn("Aarav", report["by_user"])

        mixed = (
            "[15/11/26, 09:12:05] Sam: awesome build\n"
            "15/11/2026, 9:30 pm - Alex: slow timeout error\n"
        )
        parsed = parse_chat_records(mixed)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["user"], "Sam")
        self.assertEqual(parsed[1]["user"], "Alex")


if __name__ == "__main__":
    unittest.main()
