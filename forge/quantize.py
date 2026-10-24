"""Quantization pour l'inférence (leçon 5.3) : mesurer ce qu'on gagne en mémoire et ce qu'on perd en qualité.

Ici, bitsandbytes (8 bits LLM.int8, 4 bits NF4) : quantization « à la volée » au chargement, sans données
de calibration, disponible partout où tourne PyTorch + CUDA. Les formats de déploiement (GGUF pour
llama.cpp, AWQ et GPTQ pour vLLM) se produisent avec leurs outils dédiés : voir la leçon.
"""

from __future__ import annotations

import time

import torch

from . import config, infer, metrics
from .sft import compute_dtype
from .tickets import Example


def load_quantized(model_dir: str, bits: int | None):
    from transformers import BitsAndBytesConfig

    if bits is None:
        return infer.load(model_dir)
    if bits == 8:
        q = BitsAndBytesConfig(load_in_8bit=True)
    else:
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=compute_dtype())
    return infer.load(model_dir, dtype=compute_dtype(), quantization_config=q, device_map={"": 0})


def evaluate(model_dir: str, bits: int | None, test: list[Example]) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = load_quantized(model_dir, bits)
    footprint = model.get_memory_footprint() / 1024**2
    start = time.perf_counter()
    outputs, _ = infer.run_task(model, tokenizer, test)
    seconds = (time.perf_counter() - start) / len(test)
    report = metrics.score(f"{bits or 32} bits", test, outputs)
    del model
    return {"bits": bits or 32, "poids_mo": footprint, "ms_par_exemple": seconds * 1000,
            "exact": report.exact_match, "json": report.json_valid, "rapport": report}


def compare(model_dir: str = str(config.RUNS_DIR / "merged"), test: list[Example] | None = None) -> list[dict]:
    return [evaluate(model_dir, bits, test) for bits in (None, 8, 4)]
