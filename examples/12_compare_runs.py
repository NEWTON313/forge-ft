"""Leçons 3.3 et 5.1 — Comparer des runs AVEC leur incertitude : intervalle de confiance et test apparié.

    python examples/12_compare_runs.py runs/sft-lora runs/hp/sft-lora-seed1 runs/qlora/sft-qlora

Le premier dossier sert de référence ; chaque autre run est comparé à lui sur les mêmes 300 messages.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from forge import config, infer, metrics, tickets  # noqa: E402

if __name__ == "__main__":
    test = tickets.load_split(config.DATA_DIR, "test")
    reports = []
    for path in sys.argv[1:]:
        full = not (Path(path) / "adapter_config.json").exists()
        model, tokenizer = infer.load(path if full else config.BASE_MODEL, adapter=None if full else path)
        outputs, _ = infer.run_task(model, tokenizer, test)
        report = metrics.score(Path(path).name, test, outputs)
        low, high = metrics.wilson_interval(report.exact, report.n)
        line = f"{report.name:<24} exact {report.exact_match:.2f}  IC95 [{low:.2f} ; {high:.2f}]"
        if reports:
            only_ref, only_this, p = metrics.mcnemar(reports[0], report)
            line += f"   vs {reports[0].name} : {only_ref} / {only_this} désaccords, p = {p:.3f}"
        print(line)
        reports.append(report)
        del model
        torch.cuda.empty_cache()
