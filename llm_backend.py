# /workspace/icdd-rag-pipeline/llm_backend.py
from __future__ import annotations
from typing import List, Dict
from pathlib import Path
import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

_MODEL = None
_TOKENIZER = None

def _load(model_id_or_path: str):
    global _TOKENIZER, _MODEL
    if _MODEL is not None:
        return
    _TOKENIZER = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)
    _MODEL = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    if _TOKENIZER.pad_token_id is None:
        _TOKENIZER.pad_token_id = _TOKENIZER.eos_token_id

def call_llm(messages, model_id_or_path: str, max_new_tokens: int = 256,
             temperature: float = 0.0, do_sample: bool = False, seed: int = 42):
    torch.manual_seed(seed)
    _load(model_id_or_path)

    # Use the model’s native chat template (works for Qwen2.5, Llama-2/3, Mistral-Instruct, etc.)
    prompt = _TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = _TOKENIZER(prompt, return_tensors="pt").to(_MODEL.device)
    gen = _MODEL.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,          # <- if False, ignore temp/top_p/top_k
        temperature=(None if not do_sample else temperature),
        top_p=(None if not do_sample else 0.8),
        top_k=(None if not do_sample else 20),
        pad_token_id=_TOKENIZER.pad_token_id,
        eos_token_id=_TOKENIZER.eos_token_id,
    )
    out = _TOKENIZER.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return out.strip()
