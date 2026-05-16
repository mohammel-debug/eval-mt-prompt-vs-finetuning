import os
from openai import OpenAI
from datasets import load_dataset
from huggingface_hub import login
import evaluate
from tqdm import tqdm

# Login
login('YOUR_HF_TOKEN')

# Load FLORES-200
flores_en = load_dataset("openlanguagedata/flores_plus", "eng_Latn")
flores_eu = load_dataset("openlanguagedata/flores_plus", "eus_Latn")

# OpenAI client
client = OpenAI(api_key="YOUR_OPENAI_KEY

# Metrics
metric_bleu = evaluate.load("sacrebleu")
metric_chrf = evaluate.load("chrf")

predictions = []
references = []

for i, (en, eu) in enumerate(tqdm(zip(flores_en["devtest"]["text"], flores_eu["devtest"]["text"]), total=len(flores_en["devtest"]["text"]), desc="Translating")):
    response = client.chat.completions.create(
        model="gpt-5.5",
        messages=[
            {"role": "system", "content": "You are a translator. Translate the following English text to Basque. Output only the translation, nothing else."},
            {"role": "user", "content": en}
        ]
    )
    translation = response.choices[0].message.content.strip()
    predictions.append(translation)
    references.append([eu])

    if (i + 1) % 50 == 0:
        print(f"[{i+1}] EN: {en[:60]}...")
        print(f"       EU: {translation[:60]}...")

bleu = metric_bleu.compute(predictions=predictions, references=references)
chrf = metric_chrf.compute(predictions=predictions, references=references)

print(f"\nGPT-5.5 BLEU: {bleu['score']:.2f}")
print(f"GPT-5.5 chrF: {chrf['score']:.2f}")
