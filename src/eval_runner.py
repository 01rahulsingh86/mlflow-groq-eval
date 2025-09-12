import os
import argparse
import pandas as pd
import mlflow
from tqdm import tqdm
from metrics import exact_match, contains, token_f1
from groq_pyfunc import GroqLLM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', type=str, required=True, help='CSV with columns: prompt, reference')
    ap.add_argument('--experiment', type=str, default='groq-eval-demo')
    ap.add_argument('--output_dir', type=str, default='runs/demo')
    ap.add_argument('--temperature', type=float, default=0.0)
    ap.add_argument('--max_tokens', type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    mlflow.set_experiment(args.experiment)

    df = pd.read_csv(args.dataset)
    if not {'prompt', 'reference'}.issubset(df.columns):
        raise ValueError('Dataset must have columns: prompt, reference')

    model = GroqLLM()
    # Fake MLflow context load to set api and client
    model.load_context(None)

    with mlflow.start_run():
        mlflow.log_param('provider', 'groq')
        mlflow.log_param('model', os.environ.get('GROQ_MODEL', 'llama-3.1-8b-instant'))
        mlflow.log_param('temperature', args.temperature)
        mlflow.log_param('max_tokens', args.max_tokens)

        preds = []
        for p in tqdm(df['prompt'].tolist(), desc='Querying Groq'):
            preds.append(model._infer_one(p, temperature=args.temperature, max_tokens=args.max_tokens))

        df_out = df.copy()
        df_out['prediction'] = preds
        out_csv = os.path.join(args.output_dir, 'predictions.csv')
        df_out.to_csv(out_csv, index=False)

        em = exact_match(preds, df['reference'].tolist())
        ct = contains(preds, df['reference'].tolist())
        f1 = token_f1(preds, df['reference'].tolist())

        mlflow.log_metric('exact_match', em)
        mlflow.log_metric('contains', ct)
        mlflow.log_metric('token_f1', f1)
        mlflow.log_artifact(out_csv)

        print('\n=== Metrics ===')
        print(f'Exact Match : {em:.3f}')
        print(f'Contains    : {ct:.3f}')
        print(f'Token F1    : {f1:.3f}')
        print(f'Artifacts saved to: {args.output_dir}')

if __name__ == '__main__':
    main()
