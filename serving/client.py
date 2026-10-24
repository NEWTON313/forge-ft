"""Leçon 5.4 — Interroger le serveur vLLM avec le client OpenAI (pip install openai).

    python -m serving.client        # depuis la racine du projet, serveur lancé (serving/serve.sh)
"""

from openai import OpenAI

from forge.config import SYSTEM_PROMPT

message = "Bonjour, la lampe Halo ne s'allume plus du tout, est-ce pris en charge ?"
client = OpenAI(base_url="http://localhost:8000/v1", api_key="inutile-en-local")
response = client.chat.completions.create(
    model="tri",  # le nom de l'adaptateur : le même serveur sert plusieurs spécialisations
    messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": message}],
    temperature=0, max_tokens=64,
)
print(response.choices[0].message.content)
