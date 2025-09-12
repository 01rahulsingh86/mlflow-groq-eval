# MLflow + Groq Evaluation Demo

A minimal, runnable example that evaluates an LLM hosted by **Groq** using **MLflow**.
It wraps Groq as an `mlflow.pyfunc` model, runs a small evaluation set, and logs metrics + artifacts.

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
export GROQ_API_KEY="sk-..."
python src/eval_runner.py --dataset data/eval.csv --experiment groq-eval-demo --output_dir runs/demo
mlflow ui --port 5000
```
