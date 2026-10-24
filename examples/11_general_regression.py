"""Leçon 5.1 — Le fine-tuning a-t-il dégradé les capacités générales ? Base contre modèles fine-tunés.

    python examples/11_general_regression.py runs/sft-lora runs/hp/sft-full-epochs1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from forge import config, general, infer  # noqa: E402

if __name__ == "__main__":
    candidates = [("base", None)] + [(Path(p).name, p) for p in sys.argv[1:]]
    for name, path in candidates:
        full = path and not (Path(path) / "adapter_config.json").exists()
        model, tokenizer = infer.load(path if full else config.BASE_MODEL, adapter=None if full else path)
        accuracy, as_json, outputs = general.general_score(model, tokenizer)
        print(f"{name:<22} bonnes réponses {accuracy:.2f}   réponses en JSON {as_json:.2f}")
        print(f"   « {general.QUESTIONS[0][0]} » → {outputs[0].strip()[:100]!r}")
        del model
        torch.cuda.empty_cache()
