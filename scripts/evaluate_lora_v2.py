import torch
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from datasets import load_dataset
from huggingface_hub import login
import evaluate
from tqdm import tqdm

login('YOUR_HF_TOKEN')

model_checkpoint = "Helsinki-NLP/opus-mt-en-eu"
finetuned_path   = "./results/lora_v2/opus-mt-en-eu-lora-v2/checkpoint-9375"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
device = "cuda" if torch.cuda.is_available() else "cpu"

flores_en = load_dataset("openlanguagedata/flores_plus", "eng_Latn")
flores_eu = load_dataset("openlanguagedata/flores_plus", "eus_Latn")
sources = flores_en["devtest"]["text"]
refs    = flores_eu["devtest"]["text"]

metric_bleu = evaluate.load("sacrebleu")
metric_chrf = evaluate.load("chrf")

base = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model = PeftModel.from_pretrained(base, finetuned_path)
model.eval().to(device)

predictions = []
batch_size = 32
for i in tqdm(range(0, len(sources), batch_size), desc="LoRA v2"):
    batch = sources[i:i+batch_size]
    inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                       max_length=128, padding=True).to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_length=128, num_beams=1)
    predictions.extend(tokenizer.batch_decode(output, skip_special_tokens=True))

bleu = metric_bleu.compute(predictions=predictions, references=[[r] for r in refs])
chrf = metric_chrf.compute(predictions=predictions, references=[[r] for r in refs])

print(f"LoRA v2 BLEU: {bleu['score']:.2f}")
print(f"LoRA v2 chrF: {chrf['score']:.2f}")

json.dump({"predictions": predictions, "references": list(refs),
           "bleu": bleu["score"], "chrf": chrf["score"]},
          open("results/lora_v2/lora_v2_results.json", "w"), indent=2)

