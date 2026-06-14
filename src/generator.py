"""LLM text generation via HuggingFace Inference API (no local GPU required)."""

from __future__ import annotations

import os

from huggingface_hub import InferenceClient

_DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question based ONLY on the provided context. "
    "If the context does not contain enough information, say so clearly. Do not make up facts."
)


class Generator:
    def __init__(self, model: str = _DEFAULT_MODEL, hf_token: str | None = None) -> None:
        self.model = model
        token = hf_token or os.getenv("HF_TOKEN", "")
        if not token:
            raise ValueError(
                "HuggingFace token required. Set HF_TOKEN env variable or pass hf_token=."
                "\nGet a free token at https://huggingface.co/settings/tokens"
            )
        self._client = InferenceClient(model=model, token=token)

    def generate(self, question: str, context_chunks: list[dict], max_new_tokens: int = 512) -> str:
        context = "\n\n---\n\n".join(c["text"] for c in context_chunks)
        user_message = f"Context:\n{context}\n\nQuestion: {question}"

        response = self._client.chat_completion(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_new_tokens,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
