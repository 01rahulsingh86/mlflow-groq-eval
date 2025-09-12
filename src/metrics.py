from typing import List
import re

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def exact_match(preds: List[str], refs: List[str]) -> float:
    n = len(refs)
    correct = sum(1 for p, r in zip(preds, refs) if normalize_text(p) == normalize_text(r))
    return correct / n if n else 0.0

def contains(preds: List[str], refs: List[str]) -> float:
    n = len(refs)
    correct = 0
    for p, r in zip(preds, refs):
        p2, r2 = normalize_text(p), normalize_text(r)
        if r2 and r2 in p2:
            correct += 1
    return correct / n if n else 0.0

def token_f1(preds: List[str], refs: List[str]) -> float:
    def f1(a: str, b: str) -> float:
        a_set = set(normalize_text(a).split())
        b_set = set(normalize_text(b).split())
        if not a_set or not b_set:
            return 0.0
        tp = len(a_set & b_set)
        prec = tp / len(a_set) if a_set else 0.0
        rec = tp / len(b_set) if b_set else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    scores = [f1(p, r) for p, r in zip(preds, refs)]
    return sum(scores) / len(scores) if scores else 0.0
