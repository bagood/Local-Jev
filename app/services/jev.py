import os
import string
import threading
from functools import lru_cache
from typing import Any

import numpy as np
from fastapi import HTTPException, status
from llama_cpp import Llama

from app.schemas.inference import ChoiceScore, InferenceRequest, InferenceResponse


class JevService:
    """Score choices using the local model's next-token logits."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._model_lock = threading.Lock()

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = Llama.from_pretrained(
                repo_id=os.getenv("JEV_MODEL_REPO", "Qwen/Qwen3-4B-GGUF"),
                filename=os.getenv("JEV_MODEL_FILE", "Qwen3-4B-Q8_0.gguf"),
                n_ctx=int(os.getenv("JEV_N_CTX", "512")),
                logits_all=True,
                verbose=False,
            )
        return self._model

    @staticmethod
    def _build_prompt(request: InferenceRequest, labels: list[str]) -> str:
        options = "\n".join(
            f"{label}. {choice}"
            for label, choice in zip(labels, request.choices, strict=True)
        )
        return f"""<|im_start|>system
{request.system_prompt}\n\nChoose one option.<|im_end|>
<|im_start|>user
{request.query}\n\n{options}<|im_end|>
<|im_start|>assistant
<think>\n\n</think>\n\n"""

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        labels = list(string.ascii_uppercase[: len(request.choices)])
        prompt = self._build_prompt(request, labels)

        try:
            with self._model_lock:
                model = self._get_model()
                prompt_tokens = model.tokenize(
                    prompt.encode("utf-8"), add_bos=False, special=True
                )
                model.reset()
                model.eval(prompt_tokens)
                logits = model.scores[model.n_tokens - 1]

                token_ids = []
                for label in labels:
                    label_tokens = model.tokenize(label.encode("utf-8"), add_bos=False)
                    if len(label_tokens) != 1:
                        raise RuntimeError(f"Label {label!r} is not a single token")
                    token_ids.append(label_tokens[0])

                choice_logits = np.asarray(
                    [logits[token_id] for token_id in token_ids], dtype=np.float64
                )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"The prompt could not be evaluated: {exc}",
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Local model inference failed: {exc}",
            ) from exc

        shifted = choice_logits - np.max(choice_logits)
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        scores = [
            ChoiceScore(choice=choice, probability=float(probability))
            for choice, probability in zip(
                request.choices, probabilities, strict=True
            )
        ]
        prediction = scores[int(np.argmax(probabilities))].choice
        return InferenceResponse(predicted_choice=prediction, scores=scores)


@lru_cache
def get_jev_service() -> JevService:
    return JevService()
