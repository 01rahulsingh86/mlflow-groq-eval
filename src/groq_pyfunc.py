import os
import mlflow.pyfunc
from typing import List, Any
from groq import Groq

SYSTEM_DEFAULT = "You are a helpful assistant. Answer briefly and accurately."

class GroqLLM(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY env var is not set.")
        self.model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
        self.client = Groq(api_key=self.api_key)
        self.system_prompt = os.environ.get("GROQ_SYSTEM_PROMPT", SYSTEM_DEFAULT)

    def _infer_one(self, prompt: str, temperature: float = 0.0, max_tokens: int = 256) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    def predict(self, context, model_input: Any) -> List[str]:
        outputs: List[str] = []
        if isinstance(model_input, list):
            for p in model_input:
                outputs.append(self._infer_one(str(p)))
            return outputs
        try:
            import pandas as pd
            if isinstance(model_input, pd.DataFrame):
                prompts = model_input["prompt"].tolist()
            else:
                prompts = list(model_input)
            for p in prompts:
                outputs.append(self._infer_one(str(p)))
            return outputs
        except Exception:
            for p in model_input:
                outputs.append(self._infer_one(str(p)))
            return outputs
