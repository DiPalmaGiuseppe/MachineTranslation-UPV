# %%
import os
import whisper
import pandas as pd
import time
from evaluate import load
from opencc import OpenCC
from whisper.normalizers.basic import BasicTextNormalizer

normalizer = BasicTextNormalizer()
cc = OpenCC('t2s')  # traditional -> simplified Chinese

model_base = whisper.load_model("base")
model_medium = whisper.load_model("medium")
model_large = whisper.load_model("large")

print("Whisper model device:", next(model_base.parameters()).device)

bleu_metric = load("sacrebleu")
comet_metric = load("comet")

# %%
from datasets import load_dataset
raw_datasets = load_dataset("fixie-ai/covost2", "zh-CN_en")
print(raw_datasets)

# %%
# data_to_evaluate = raw_datasets["validation"].select(range(10))
data_to_evaluate = raw_datasets["validation"]

def evaluate_model(model, model_name, dataset):
    
    print(f"Evaluating model: {model_name}")
    
    sample = {
        "hypothesis": [],
        "translation": [],
        "sentence": [],
        "training_time": 0
    }

    t_start = time.time()
    for s in dataset:
        audio_array = s["audio"]["array"].astype("float32")
        ref = s["sentence"]
        
        # Translation
        result = model.transcribe(audio_array, language="zh", task="translate")
        hyp = result['text']
        
        sample["hypothesis"].append(hyp.strip())
        sample["sentence"].append(ref.strip())
        sample["translation"].append(s["translation"].strip())
        
    sample["training_time"] = time.time() - t_start
        
    return sample


# %%
sample_base = evaluate_model(model_base, "base", data_to_evaluate)
sample_medium = evaluate_model(model_medium, "medium", data_to_evaluate)
sample_large = evaluate_model(model_large, "large", data_to_evaluate)
samples = [("base", sample_base), ("medium", sample_medium), ("large", sample_large)]

for model_name, sample in samples:
    print(f"\n--- Analysis for Whisper {model_name} ---")
    
    print(f"Training time: {sample['training_time']:.2f} seconds")
    
    sample["clean_sentence"] = [normalizer(text).strip() for text in sample["sentence"]]
    sample["clean_hypothesis"] = [normalizer(text).strip() for text in sample["hypothesis"]]
    sample["clean_translation"] = [normalizer(text).strip() for text in sample["translation"]]

    # BLEU and COMET
    bleu = bleu_metric.compute(
        predictions=sample["hypothesis"],
        references=sample["translation"]
    )
    comet = comet_metric.compute(
        predictions=sample["hypothesis"],
        references=sample["translation"],
        sources=sample["sentence"]
    )

    # BLEU and COMET on cleaned text
    bleu_clean = bleu_metric.compute(
        predictions=sample["clean_hypothesis"],
        references=sample["clean_translation"]
    )
    comet_clean = comet_metric.compute(
        predictions=sample["clean_hypothesis"],
        references=sample["clean_translation"],
        sources=sample["clean_sentence"]
    )
    
    print(f"BLEU: {bleu['score']:.2f}")
    print(f"COMET: {comet['mean_score'] * 100:.2f}")
    print(f"BLEU (cleaned): {bleu_clean['score']:.2f}")
    print(f"COMET (cleaned): {comet_clean['mean_score'] * 100:.2f}")  
    
    column_order = [
        "sentence",
        "hypothesis",
        "translation",
        "clean_sentence",
        "clean_hypothesis",
        "clean_translation",
        "training_time"
    ]
    
    dataframe = pd.DataFrame(sample)[column_order]
    pd.set_option('display.max_colwidth', None)
    if os.path.exists('ST_csvs') == False:
        os.mkdir('ST_csvs')
    dataframe.to_csv(f'ST_csvs/ST_{model_name}.csv', encoding='utf-8', index=False)


