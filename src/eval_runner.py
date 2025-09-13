import os
import argparse
import pandas as pd
import mlflow
from tqdm import tqdm
from metrics import exact_match, contains, token_f1
from groq_pyfunc import GroqLLM
from judge import make_client, judge_one  # keep if you added judge.py

# --- DagsHub setup (optional) ---
try:
    from dagshub import init as dagshub_init
    dagshub_init(
        repo_owner="01rahulsingh86",
        repo_name="mlflow-groq-eval",
        mlflow=True
    )
except Exception as e:
    print("DagsHub init skipped:", e)
# --------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, required=True, help='CSV with columns: prompt, reference')
    ap.add_argument('--experiment', type=str, default='groq-eval-demo')
    ap.add_argument('--output_dir', type=str, default='runs/demo')
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--max_tokens', type=int, default=256)
    ap.add_argument('--judge', action='store_true', help='Enable LLM-as-judge metrics (coherence, relevance, groundedness)')
    ap.add_argument('--judge_model', type=str, default=os.environ.get('GROQ_JUDGE_MODEL', 'llama-3.1-8b-instant'))
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    mlflow.set_experiment(args.experiment)

    df = pd.read_csv(args.dataset)
    if not {'prompt', 'reference'}.issubset(df.columns):
        raise ValueError('Dataset must have columns: prompt, reference')

    # Init Groq model wrapper
    model = GroqLLM()
    model.load_context(None)  # sets API key, client, model name from env

    with mlflow.start_run():
        # Params
        mlflow.log_param('provider', 'groq')
        mlflow.log_param('model', os.environ.get('GROQ_MODEL', 'llama-3.1-8b-instant'))
        mlflow.log_param('temperature', args.temperature)
        mlflow.log_param('max_tokens', args.max_tokens)
        mlflow.log_param('judge_enabled', bool(args.judge))
        if args.judge:
            mlflow.log_param('judge_model', args.judge_model)

        # Predictions
        preds = []
        for p in tqdm(df['prompt'].tolist(), desc='Querying Groq'):
            preds.append(model._infer_one(p, temperature=args.temperature, max_tokens=args.max_tokens))

        # Save per-row outputs
        df_out = df.copy()
        df_out['prediction'] = preds
        out_csv = os.path.join(args.output_dir, 'predictions.csv')
        df_out.to_csv(out_csv, index=False)
        mlflow.log_artifact(out_csv)

        # Base metrics
        em = exact_match(preds, df['reference'].tolist())
        ct = contains(preds, df['reference'].tolist())
        f1 = token_f1(preds, df['reference'].tolist())
        mlflow.log_metric('exact_match', em)
        mlflow.log_metric('contains', ct)
        mlflow.log_metric('token_f1', f1)

        # Judge metrics (optional)
        if args.judge:
            jclient = make_client()
            coh = rel = gro = 0.0
            judge_rows = []
            for pmt, pred, ref in zip(df_out['prompt'], df_out['prediction'], df_out['reference']):
                try:
                    j = judge_one(jclient, args.judge_model, prompt=pmt, prediction=pred, reference=str(ref))
                except Exception as je:
                    j = {"coherence": 0.0, "relevance": 0.0, "groundedness": 0.0, "rationale": f"judge error: {je}"}
                judge_rows.append({"prompt": pmt, "prediction": pred, "reference": ref, **j})
                coh += j["coherence"]; rel += j["relevance"]; gro += j["groundedness"]
            n = max(1, len(df_out))
            coh /= n; rel /= n; gro /= n
            mlflow.log_metric('coherence', coh)
            mlflow.log_metric('relevance', rel)
            mlflow.log_metric('groundedness', gro)

            # Save per-row judgments
            import json
            jpath = os.path.join(args.output_dir, 'judgments.jsonl')
            with open(jpath, 'w', encoding='utf-8') as f:
                for row in judge_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
            mlflow.log_artifact(jpath)

        # Console summary
        print('\n=== Metrics ===')
        print(f'Exact Match : {em:.3f}')
        print(f'Contains    : {ct:.3f}')
        print(f'Token F1    : {f1:.3f}')
        if args.judge:
            print(f'Coherence   : {coh:.3f}')
            print(f'Relevance   : {rel:.3f}')
            print(f'Groundedness: {gro:.3f}')
        print(f'Artifacts saved to: {args.output_dir}')

if __name__ == '__main__':
    main()
