import os
import subprocess
import pandas as pd
from metrics import exact_match, contains, token_f1  # from src/
# Optional: judge aggregator if you enabled --judge

DATASET = "data/eval.csv"
OUTDIR  = "runs/ci"

def test_llm_quality():
    os.makedirs(OUTDIR, exist_ok=True)
    # run the evaluator deterministically
    cmd = [
        "python", "src/eval_runner.py",
        "--dataset", DATASET,
        "--experiment", "ci-eval",
        "--output_dir", OUTDIR,
        "--temperature", "0.0",
        "--max_tokens", "16",
        # "--judge",  # enable if you added judge.py and env vars
    ]
    subprocess.run(cmd, check=True)

    # recompute metrics from the emitted CSV to assert thresholds
    df = pd.read_csv(f"{OUTDIR}/predictions.csv")
    preds = df["prediction"].tolist()
    refs  = df["reference"].tolist()

    em  = exact_match(preds, refs)
    ct  = contains(preds, refs)
    f1  = token_f1(preds, refs)

    assert em >= 0.70, f"ExactMatch too low: {em:.3f}"
    assert f1 >= 0.80, f"TokenF1 too low: {f1:.3f}"
    assert ct >= 0.95, f"Contains too low: {ct:.3f}"

    # if using judge:
    # import json
    # with open(f"{OUTDIR}/judgments.jsonl") as f:
    #     rows = [json.loads(l) for l in f]
    # coh = sum(r["coherence"] for r in rows)/len(rows)
    # rel = sum(r["relevance"] for r in rows)/len(rows)
    # gro = sum(r["groundedness"] for r in rows)/len(rows)
    # assert coh >= 0.9 and rel >= 0.9 and gro >= 0.9
