# %%
import whisper
import pandas as pd
import jiwer
import jieba
import re
from opencc import OpenCC
from whisper.normalizers.basic import BasicTextNormalizer

normalizer = BasicTextNormalizer()
cc = OpenCC('t2s')  # traditional -> simplified Chinese

model_base = whisper.load_model("base")
model_medium = whisper.load_model("medium")
model_large = whisper.load_model("large")

print("Whisper model device:", next(model_base.parameters()).device)



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
        "hypothesis_simple": [],
        "hypothesis_traditional": [],
        "sentence": []
    }

    for s in dataset:
        audio_array = s["audio"]["array"].astype("float32")
        ref = s["sentence"]
        
        # ASR
        result = model.transcribe(audio_array, language="zh")
        hyp_tr = result['text']
        
        # Conversione e pulizia
        hyp_simp = cc.convert(hyp_tr).strip()
        ref_clean = ref.strip()
        
        sample["hypothesis_simple"].append(hyp_simp)
        sample["hypothesis_traditional"].append(hyp_tr)
        sample["sentence"].append(ref_clean)
        
    return sample


# %%
def word_tokenize_zh(text):
    text = normalizer(text)
    text = re.sub(r'[^\w\s]', '', text)
    words = jieba.cut(text)
    return " ".join(words)

# %%
sample_base = evaluate_model(model_base, "base", data_to_evaluate)
sample_medium = evaluate_model(model_medium, "medium", data_to_evaluate)
sample_large = evaluate_model(model_large, "large", data_to_evaluate)
samples = [("base", sample_base), ("medium", sample_medium), ("large", sample_large)]

for model_name, sample in samples:
    print(f"\n--- Analysis for Whisper {model_name} ---")
    
    # Calcolo CER (quello che hai già fatto, carattere per carattere)
    refs_char = [" ".join(list(normalizer(text))) for text in sample["sentence"]]
    hyps_tr_char = [" ".join(list(normalizer(text))) for text in sample["hypothesis_traditional"]]
    hyps_simp_char = [" ".join(list(normalizer(text))) for text in sample["hypothesis_simple"]]
    cer_tr = jiwer.wer(refs_char, hyps_tr_char)
    cer_simp = jiwer.wer(refs_char, hyps_simp_char)
    
    # Calcolo WER (usando il tokenizzatore jieba)
    refs_word = [word_tokenize_zh(text) for text in sample["sentence"]]
    hyps_tr_word = [word_tokenize_zh(text) for text in sample["hypothesis_traditional"]]
    hyps_simp_word = [word_tokenize_zh(text) for text in sample["hypothesis_simple"]]
    wer_tr = jiwer.wer(refs_word, hyps_tr_word)
    wer_simp = jiwer.wer(refs_word, hyps_simp_word)

    print(f"CER (Character-level Traditional): {cer_tr*100:.2f} %")
    print(f"CER (Character-level Simplified): {cer_simp*100:.2f} %")
    print(f"WER (Word-level Traditional with Jieba): {wer_tr*100:.2f} %")
    print(f"WER (Word-level Simplified with Jieba): {wer_simp*100:.2f} %")
# dataframe = pd.DataFrame(sample)
# pd.set_option('display.max_colwidth', None)
# dataframe.to_csv('L4.1_ASR_Whisper_Baseline_dev_Covost2.csv', encoding='utf-8', index=False)



