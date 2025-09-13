# MLflow + Groq Evaluation Demo

A minimal, runnable example that evaluates an LLM hosted by **Groq** using **MLflow**.
It wraps Groq as an `mlflow.pyfunc` model, runs a small evaluation set, and logs metrics + artifacts.
You have a tiny list of questions + correct answers in data/eval.csv.
The code calls Groq’s LLM for each question to get a predicted answer.
It then compares prediction vs. the expected answer with simple checks:
Exact match (identical text),
Contains (predicted text includes the expected text),
Token-level F1 (word overlap score).
All results (metrics + the CSV of predictions) are logged to MLflow so you can see runs, compare them, and keep history.
So it’s basically: ask model → score answers → store scores & outputs somewhere nice (MLflow).

## Prereqs
- Python 3.9+
- A **Groq API key** (env var `GROQ_API_KEY`)
- Optional model override: `GROQ_MODEL` (default: `llama-3.1-8b-instant`)

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
Use the dagshub client (convenient)
pip install dagshub

from dagshub import init
init(repo_owner="<OWNER>", repo_name="<REPO>", mlflow=True)
# This configures MLFLOW_TRACKING_URI and credentials for you (with your token).

# plus your Groq creds
export GROQ_API_KEY="sk-..."
python src/eval_runner.py --dataset data/eval.csv --experiment groq-eval-demo --output_dir runs/demo

# Recommended
export MLFLOW_TRACKING_USERNAME="<your_dagshub_username>"
export MLFLOW_TRACKING_PASSWORD="<your_dagshub_access_token>"

# (Alternatively, some setups use)
# export DAGSHUB_USERNAME="<your_dagshub_username>"
# export DAGSHUB_TOKEN="<your_dagshub_access_token>"

Edit src/eval_runner.py

Tip: temperature=0.0 reduces randomness, which is best for consistent, comparable evaluations.

Add this block near the imports, above where you call MLflow:


# --- DagsHub setup (put this near the top) ---
try:
    from dagshub import init as dagshub_init
    dagshub_init(
        repo_owner="01rahulsingh86",          # <OWNER>
        repo_name="mlflow-groq-eval",         # <REPO>
        mlflow=True                           # configures MLflow tracking URI & creds
    )
except Exception as e:
    print("DagsHub init skipped:", e)
# --------------------------------------------

export GROQ_API_KEY="sk-..."
python src/eval_runner.py --dataset data/eval.csv --experiment groq-eval-demo --output_dir runs/demo
mlflow ui --port 5000

export GROQ_API_KEY="sk-..."

# Try with diff models

# 8B
export GROQ_MODEL="llama-3.1-8b-instant"
python src/eval_runner.py --dataset data/eval.csv --experiment groq-eval --output_dir runs/8b

# 70B
export GROQ_MODEL="llama-3.1-70b-versatile"
python src/eval_runner.py --dataset data/eval.csv --experiment groq-eval --output_dir runs/70b

# Mixtral
export GROQ_MODEL="mixtral-8x7b"
python src/eval_runner.py --dataset data/eval.csv --experiment groq-eval --output_dir runs/mixtral

#Metric Defination
Metric definitions (simple & fast)

contains
Fraction of rows where normalize(prediction) includes normalize(reference) as a substring.
Useful for “did it say the key word/number somewhere?”

exact_match
Fraction of rows where normalized strings are exactly equal.
Strict: good to detect perfectly formatted answers.

token_f1
Average F1 over token sets (word overlap).
More forgiving than exact match; penalizes missing/extra words.

By default normalization is light; you can make it stricter (e.g., strip punctuation) in metrics.py to turn “Paris.” and “Paris” into a match for F1.


# run llm as judge command
python src/eval_runner.py \           
  --dataset data/eval.csv \
  --experiment groq-eval-demo \
  --output_dir runs/with-judge-2 \
  --temperature 0.0 \
  --judge

#run only tests
pytest -q   

#running with temperature 

python src/eval_runner.py --dataset data/eval.csv --experiment groq-eval --output_dir runs/t02 --temperature 1

#Grok Key Export to be done way before running anything
export GROQ_API_KEY=""
export GROQ_JUDGE_MODEL="


```
