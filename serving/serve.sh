#!/usr/bin/env bash
# Leçon 5.4 — Servir Forge avec vLLM (Linux, GPU NVIDIA de capacité de calcul 7.0 ou plus). Non exécuté sur le
# matériel du cours (GTX 1060 sous Windows) : vérifiez les options avec `vllm serve --help` de votre version.

# Modèle fusionné (adaptateur LoRA intégré aux poids, leçon 5.3)
vllm serve ./runs/merged --served-model-name nimbus-tri --max-model-len 2048

# Ou : modèle de base + adaptateurs LoRA chargés à côté, choisis requête par requête
vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-lora \
    --lora-modules tri=./runs/sft-lora reponses=./runs/dpo-sigmoid-0.1 --max-lora-rank 16
