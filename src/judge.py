# src/judge.py
import os, json, time, traceback, re
from typing import Optional, Dict, Any
from groq import Groq

SYSTEM = (
  "You are a meticulous evaluation judge. "
  "Return ONLY valid compact JSON with float scores in [0,1] and a short 'rationale'. "
  "Do not include any text before or after the JSON."
)

RUBRIC = """
Evaluate the MODEL_ANSWER against the PROMPT{maybe_ref} on:
- coherence: Is the answer logically consistent and understandable?
- relevance: Does it address the user's request without going off-topic?
- groundedness: {grounded_desc}

Return JSON exactly:
{"coherence":0-1,"relevance":0-1,"groundedness":0-1,"rationale":"<brief why>"}
"""

def _rubric_text(has_ref: bool) -> str:
    grounded_desc = (
        "Is the answer supported by the REFERENCE? Penalize contradictions or invented facts."
        if has_ref else
        "No reference provided; grade groundedness as 1.0 if the answer makes no unsupported claims."
    )
    return RUBRIC.format(
        maybe_ref=" and REFERENCE" if has_ref else "",
        grounded_desc=grounded_desc
    )

def make_client() -> Groq:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY missing")
    return Groq(api_key=key)

def _call_with_retries(client: Groq, payload: dict, retries: int = 2, backoff: float = 0.7):
    last_exc = None
    for i in range(retries + 1):
        try:
            return client.chat.completions.create(**payload)
        except Exception as e:
            last_exc = e
            if i < retries:
                time.sleep(backoff * (2 ** i))
            else:
                raise last_exc

def judge_one(
    client: Groq,
    model: str,
    prompt: str,
    prediction: str,
    reference: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 200
) -> Dict[str, Any]:
    has_ref = bool(reference and str(reference).strip())
    rubric = _rubric_text(has_ref)
    user = (
        f"PROMPT:\n{prompt}\n\n"
        f"MODEL_ANSWER:\n{prediction}\n\n" +
        (f"REFERENCE:\n{reference}\n\n" if has_ref else "") +
        "Score now."
    )
    payload = dict(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": rubric + "\n\n" + user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    try:
        resp = _call_with_retries(client, payload)
        text = resp.choices[0].message.content.strip()
        try:
            out = json.loads(text)
        except Exception:
            m = re.search(r"\{.*\}", text, re.S)
            out = json.loads(m.group(0)) if m else {"coherence":0,"relevance":0,"groundedness":0,"rationale": text[:300]}
    except Exception as e:
        # Capture full traceback for later debugging
        tb = traceback.format_exc()
        return {
            "coherence": 0.0,
            "relevance": 0.0,
            "groundedness": 0.0,
            "rationale": f"judge error: {repr(e)} | {tb.splitlines()[-1][:200]}"
        }

    def clamp01(x):
        try:
            v = float(x)
            return 0.0 if v < 0 else 1.0 if v > 1 else v
        except Exception:
            return 0.0

    return {
        "coherence": clamp01(out.get("coherence", 0)),
        "relevance": clamp01(out.get("relevance", 0)),
        "groundedness": clamp01(out.get("groundedness", 0)),
        "rationale": str(out.get("rationale","")).strip()[:500]
    }
    