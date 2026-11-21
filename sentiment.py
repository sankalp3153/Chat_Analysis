"""
Chat Sentiment & Toxicity Analysis Engine (English + Hinglish support).

Scores individual messages into polarity (-1.0..+1.0), categorical sentiment
('positive', 'neutral', 'negative'), and flags toxic/cyber-troll expressions.
"""

from collections import defaultdict
from dataclasses import dataclass
import re
from typing import Dict, Iterable, List

_TOKEN_RE = re.compile(r"[a-zA-Z']+")

POSITIVE_WORDS = {
    "awesome", "great", "good", "nice", "love", "loved", "amazing", "happy", "thanks",
    "thank", "clean", "cleanly", "fixed", "better", "best", "congrats", "congratulations",
    "super", "cool", "win", "winning", "glad", "excited", "helpful", "fast", "solid",
    "brilliant", "perfect", "well", "appreciate", "yay", "smooth", "green", "mst", "mast",
    "badhiya", "badiya", "shandar", "sahi", "accha", "acha", "zabardast", "khush",
}

NEGATIVE_WORDS = {
    "bad", "worst", "hate", "hated", "frustrated", "annoying", "broken", "bug", "bugs",
    "error", "errors", "failed", "fail", "failing", "slow", "flaky", "wasted", "waste",
    "terrible", "awful", "horrible", "sad", "angry", "upset", "crash", "crashed",
    "timeout", "delay", "delayed", "stuck", "useless", "sucks", "pathetic", "stupid",
    "idiot", "trash", "bekar", "bakwas", "ganda", "kharab", "dimag", "pagal",
}

TOXIC_WORDS = {
    "idiot", "stupid", "moron", "trash", "useless", "pathetic", "shut up", "loser",
    "fool", "dumb", "troll", "bakwas", "pagal", "bewakoof",
}

NEGATORS = {"not", "no", "never", "dont", "don't", "cant", "can't", "nahi", "nhi", "mat"}


@dataclass(frozen=True)
class SentimentResult:
    score: float
    label: str  # "positive" | "neutral" | "negative"
    is_toxic: bool


def score_message(text: str) -> SentimentResult:
    if not text or not text.strip():
        return SentimentResult(score=0.0, label="neutral", is_toxic=False)

    lowered = text.lower()
    tokens = _TOKEN_RE.findall(lowered)
    if not tokens:
        return SentimentResult(score=0.0, label="neutral", is_toxic=False)

    pos = 0.0
    neg = 0.0
    toxic_hit = any(term in lowered for term in TOXIC_WORDS if " " in term)

    for idx, tok in enumerate(tokens):
        negated = idx > 0 and tokens[idx - 1] in NEGATORS
        if tok in TOXIC_WORDS:
            toxic_hit = True
        if tok in POSITIVE_WORDS:
            if negated:
                neg += 1.0
            else:
                pos += 1.0
        elif tok in NEGATIVE_WORDS:
            if negated:
                pos += 0.5
            else:
                neg += 1.0

    total = pos + neg
    if total == 0:
        return SentimentResult(score=0.0, label="neutral", is_toxic=toxic_hit)

    compound = round((pos - neg) / total, 3)
    if compound >= 0.15:
        label = "positive"
    elif compound <= -0.15:
        label = "negative"
    else:
        label = "neutral"

    return SentimentResult(score=compound, label=label, is_toxic=toxic_hit)


def analyze_records(records: Iterable[Dict[str, object]]) -> Dict[str, object]:
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    toxic_count = 0
    scores: List[float] = []
    per_user: Dict[str, List[float]] = defaultdict(list)

    for rec in records:
        user = str(rec.get("user", ""))
        if user == "group_notification":
            continue
        res = score_message(str(rec.get("message", "")))
        counts[res.label] += 1
        if res.is_toxic:
            toxic_count += 1
        scores.append(res.score)
        per_user[user].append(res.score)

    total = sum(counts.values())
    avg_score = round(sum(scores) / len(scores), 3) if scores else 0.0
    user_summary = {
        user: {
            "messages": len(u_scores),
            "avg_sentiment": round(sum(u_scores) / len(u_scores), 3),
        }
        for user, u_scores in sorted(per_user.items())
    }

    return {
        "total_messages": total,
        "average_score": avg_score,
        "distribution": counts,
        "toxic_messages": toxic_count,
        "by_user": user_summary,
    }
