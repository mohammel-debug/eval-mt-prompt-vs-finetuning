import os
from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import evaluate
from tqdm import tqdm
import json

login('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')

# Load FLORES-200
flores_en = load_dataset("openlanguagedata/flores_plus", "eng_Latn")
flores_eu = load_dataset("openlanguagedata/flores_plus", "eus_Latn")

# Load NLLB model
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Metrics
metric_bleu = evaluate.load("sacrebleu")
metric_chrf = evaluate.load("chrf")

predictions = []
references  = []

for i, (en, eu) in enumerate(tqdm(
    zip(flores_en["devtest"]["text"], flores_eu["devtest"]["text"]),
    total=len(flores_en["devtest"]["text"]),
    desc="Translating"
)):
    inputs = tokenizer(en, return_tensors="pt", truncation=True, max_length=256).to(device)
    output = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("eus_Latn"),
        max_length=256,
        num_beams=4,
    )
    translation = tokenizer.decode(output[0], skip_special_tokens=True)
    predictions.append(translation)
    references.append([eu])

    # Save every 50 sentences
    if (i + 1) % 50 == 0:
        print(f"[{i+1}] EN: {en[:60]}...")
        print(f"       EU: {translation[:60]}...")
        with open("nllb_results.json", "w") as f:
            json.dump({"predictions": predictions, "references": [r[0] for r in references]}, f)

bleu = metric_bleu.compute(predictions=predictions, references=references)
chrf = metric_chrf.compute(predictions=predictions, references=references)

print(f"\nNLLB BLEU : {bleu['score']:.2f}")
print(f"NLLB chrF : {chrf['score']:.2f}")

# Save final results
with open("nllb_results.json", "w") as f:
    json.dump({
        "predictions": predictions,
        "references" : [r[0] for r in references],
        "bleu"       : bleu["score"],
        "chrf"       : chrf["score"],
    }, f, indent=2)

print("Results saved to nllb_results.json")
