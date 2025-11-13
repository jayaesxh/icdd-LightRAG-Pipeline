# rag_llm_adapter.py
# Tiny adapter to plug your local HF LLM into LightRAG.

from __future__ import annotations
from typing import List, Optional
from llm_backend import call_llm  # you already created this in previous step

async def hf_model_complete(prompt: str,
                            system_prompt: Optional[str] = None,
                            history_messages: List[dict] = [],
                            keyword_extraction: bool = False,
                            **kwargs) -> str:
    """
    LightRAG calls this like an OpenAI-style function.
    We map it to your local model via call_llm(...).
    """
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    for h in history_messages or []:
        if h.get("role") and h.get("content"):
            msgs.append(h)
    msgs.append({"role": "user", "content": prompt})
    out = call_llm(msgs,
                   model_id_or_path=kwargs.get("llm_model_name") or kwargs.get("model_id") or "/workspace/models/qwen2.5-7b-instruct",
                   max_new_tokens=kwargs.get("max_new_tokens", 512),
                   temperature=kwargs.get("temperature", 0.2))
    return out
