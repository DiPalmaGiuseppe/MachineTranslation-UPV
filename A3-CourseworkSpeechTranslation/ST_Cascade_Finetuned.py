# %%
from evaluate import load

import gc
import math
import torch
import numpy as np
from torch.utils.data import DataLoader
from whisper.normalizers.basic import BasicTextNormalizer
from datasets import load_dataset, DatasetDict
from huggingface_hub import login
from transformers import AutoTokenizer
from transformers import GenerationConfig
from transformers import BitsAndBytesConfig
from transformers import Seq2SeqTrainer
from transformers import AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments 
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from peft import IA3Config
from evaluate import load


login(token="")

bleu_metric = load("sacrebleu")
comet_metric = load("comet")

DEBUG_MODE = False
DEBUG_FRACTION = 0.5

# %%
raw_datasets = load_dataset("csv", data_files = "ASR_csvs/ASR_large.csv")
def is_complete(example):
    for v in example.values():

        if v is None:
            return False

        if isinstance(v, float) and math.isnan(v):
            return False

        if isinstance(v, str) and v.strip() == "":
            return False

    return True
raw_datasets = raw_datasets.filter(is_complete)

if DEBUG_MODE:
    raw_datasets = DatasetDict({
        split: raw_datasets[split]
        .shuffle(seed=42)
        .select(range(int(len(raw_datasets[split]) * DEBUG_FRACTION)))
        for split in raw_datasets.keys()
    })

raw_datasets = raw_datasets["train"].train_test_split(test_size=0.3, shuffle=False)

test_valid_split = raw_datasets["test"].train_test_split(test_size=0.5, shuffle=False)

raw_datasets = DatasetDict({
    "train": raw_datasets["train"],
    "validation": test_valid_split["train"],  # metà del test diventa validation
    "test": test_valid_split["test"]          # metà del test rimane test
})

print(raw_datasets)

# %%
max_tok_length = 275
checkpoint_nllb = "facebook/nllb-200-distilled-600M"
checkpoint_mbart = "facebook/mbart-large-50-many-to-many-mmt"

src_code = "zho_Hans"
tgt_code = "eng_Latn"

src_code_mbart_trad = "zh_CN"
src_code_mbart_simple = "zh_CN"
tgt_code_mbart = "en_XX"

tokenizer_nllb = AutoTokenizer.from_pretrained(
    checkpoint_nllb, 
    padding=True, 
    pad_to_multiple_of=8, 
    src_lang=src_code, 
    tgt_lang=tgt_code, 
    truncation=False, 
    max_length=max_tok_length,
)

tokenizer_mbart_simple = AutoTokenizer.from_pretrained(
    checkpoint_mbart,
    padding=True,
    pad_to_multiple_of=8,
    src_lang=src_code_mbart_simple,
    tgt_lang=tgt_code_mbart,
    truncation=True,
    max_length=max_tok_length,
)

tokenizer_mbart_trad = AutoTokenizer.from_pretrained(
    checkpoint_mbart,
    padding=True,
    pad_to_multiple_of=8,
    src_lang=src_code_mbart_trad,
    tgt_lang=tgt_code_mbart,
    truncation=True,
    max_length=max_tok_length,
)

normalizer = BasicTextNormalizer()


# %%
def preprocess_function(sample, tokenizer, src, tgt):
    model_inputs = tokenizer(
        sample[src],
        text_target=sample[tgt],
        padding="max_length",
        truncation=True,
        max_length=max_tok_length
    )
    return model_inputs


# %%
dataset_mbart_simple = raw_datasets.map(
    lambda x: preprocess_function(x, tokenizer_mbart_simple, src="hypothesis_simple", tgt="translation"),
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)

dataset_mbart_trad = raw_datasets.map(
    lambda x: preprocess_function(x, tokenizer_mbart_trad, src="hypothesis_traditional", tgt="translation"),
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)

# %%
dataset_nllb_simple_clean = raw_datasets.map(
    lambda x: preprocess_function(x, tokenizer_nllb, src="clean_hypothesis_simple", tgt="clean_translation"),
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)
dataset_nllb_trad_clean = raw_datasets.map(
    lambda x: preprocess_function(x, tokenizer_nllb, src="clean_hypothesis_traditional", tgt="clean_translation"),
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)

# %%
def build_model(
    checkpoint,
    quantization_config,
    config = None,
):
    model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint,
        quantization_config=quantization_config,
        device_map="auto",
    )
    
    generation_config = GenerationConfig.from_pretrained(
        checkpoint,
    )
    
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=False,
        gradient_checkpointing_kwargs={'use_reentrant': False}
    )
    
    if config == "lora":
        train_config = LoraConfig(
            task_type="SEQ_2_SEQ_LM",
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
    elif config == "ia3":
        train_config = IA3Config(
            task_type="SEQ_2_SEQ_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            feedforward_modules=["fc1", "fc2"]
        )
    
    model = get_peft_model(model, train_config)
    model.print_trainable_parameters()        
    
    return model, generation_config

# %%
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# %%
def evaluate_metrics(sample, training = False):
    if training:
        inputs, labels = sample
    else:
        inputs, preds, labels = sample
        
    inputs = [normalizer(text) for text in inputs]
    preds = [normalizer(text) for text in preds]
    labels = [normalizer(text) for text in labels]
    
    bleu_result = bleu_metric.compute(predictions=preds, references=[labels[i] for i in range(len(labels))])["score"]
    results = {
        "bleu": bleu_result,
    }
    
    if not training:
        results["comet"] = comet_metric.compute(predictions=preds, references=labels, sources=inputs)["mean_score"] * 100
    
    return results

# %%
test_batch_size = 4

def get_trainer(
    model_name,
    model,
    train_dataset,
    eval_dataset,
    tokenizer,
    data_collator,
    compute_metrics
):
    training_args = Seq2SeqTrainingArguments(
        f"{model_name}-finetuned-ch-to-en",
        eval_use_gather_object="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=test_batch_size,
        per_device_eval_batch_size=test_batch_size,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=5,
        predict_with_generate=True,
        disable_tqdm=True
    )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, train=True)
    )
    
    return trainer

# %%

def evaluate_model(
    model_name,
    model, 
    generation_config, 
    dataset, 
    tokenizer,
    num_beams,
):
    model.eval()
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding='longest',
        return_tensors="pt"
    )
    
    test_dataloader = DataLoader(
        dataset["test"],
        batch_size=32,
        collate_fn=data_collator,
    )
    
    trainer = get_trainer(
        model_name=model_name,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=evaluate_metrics
    )
    
    print(" ---- Training Phase ---- ")
    trainer.train()
    
    
    print(" ---- Evaluation Phase ---- ")
    input_sequences = []
    pred_sequences = []
    label_sequences = []
    
    for batch in test_dataloader:
        with torch.no_grad():
            output_batch = model.generate(
                generation_config=generation_config, 
                input_ids=batch["input_ids"].cuda(), 
                attention_mask=batch["attention_mask"].cuda(), 
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_code), 
                max_length=max_tok_length, 
                num_beams=num_beams, 
                do_sample=False,
            )
            
        input_sequences.extend(batch["input_ids"].cpu().numpy())
        pred_sequences.extend(output_batch.cpu().numpy())
        label_sequences.extend(batch["labels"].cpu().numpy())
    
    dedoced_inputs = tokenizer.batch_decode(input_sequences, skip_special_tokens=True)
    decoded_preds = tokenizer.batch_decode(pred_sequences, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(label_sequences, skip_special_tokens=True)
    
    del trainer
    del data_collator
    del test_dataloader

    torch.cuda.empty_cache()
    gc.collect()
    
    return dedoced_inputs, decoded_preds, decoded_labels


# %%
import pandas as pd

# Creiamo lista di configurazioni con più beams
cfg_list = []

for model_name, base_cfg in {
    # "nllb_simple": (model_nllb, generation_config_nllb, dataset_nllb_simple, tokenizer_nllb),
    # "nllb_trad": (model_nllb, generation_config_nllb, dataset_nllb_trad, tokenizer_nllb),
    "mbart_simple_lora": (checkpoint_mbart, "lora", dataset_mbart_simple, tokenizer_mbart_simple,1),
    "mbart_simple_ia3": (checkpoint_mbart, "ia3", dataset_mbart_simple, tokenizer_mbart_simple,1),
    "mbart_trad_lora": (checkpoint_mbart, "lora", dataset_mbart_trad, tokenizer_mbart_trad,1),
    "mbart_trad_ia3": (checkpoint_mbart, "ia3", dataset_mbart_trad, tokenizer_mbart_trad,1),
    "nllb_simple_clean_beam1": (checkpoint_nllb, "lora", dataset_nllb_simple_clean, tokenizer_nllb, 1),
    "nllb_simple_clean_beam4": (checkpoint_nllb, "lora", dataset_nllb_simple_clean, tokenizer_nllb, 4),
    "nllb_trad_clean_beam1": (checkpoint_nllb, "lora", dataset_nllb_trad_clean, tokenizer_nllb, 1),
    "nllb_trad_clean_beam4": (checkpoint_nllb, "lora", dataset_nllb_trad_clean, tokenizer_nllb, 4),
    # "mbart_simple_clean": (model_mbart, generation_config_mbart, dataset_mbart_simple_clean, tokenizer_mbart_simple),
    # "mbart_trad_clean": (model_mbart, generation_config_mbart, dataset_mbart_trad_clean, tokenizer_mbart_trad),
}.items():
    checkpoint, config, dataset, tokenizer, num_beams = base_cfg
    cfg_list.append({
        "model_name": model_name,
        "checkpoint": checkpoint,
        "config": config,
        "dataset": dataset,
        "tokenizer": tokenizer,
        "num_beams": num_beams
    })

results_list = []

for cfg_item in cfg_list:
    name = cfg_item["model_name"]
    num_beams = cfg_item["num_beams"]
    print(f"Evaluating model: {name} with num_beams={num_beams}")
    
    model, generation_config = build_model(
        cfg_item["checkpoint"],
        quantization_config,
        cfg_item["config"],
    )

    inputs, preds, labels = evaluate_model(
        model_name=name,
        model=model,
        generation_config=generation_config,
        dataset=cfg_item["dataset"],
        tokenizer=cfg_item["tokenizer"],
        num_beams=num_beams,
    )
    
    metrics = evaluate_metrics((inputs, preds, labels))
    
    results_list.append({
        "model_name": name,
        "num_beams": num_beams,
        "bleu": metrics["bleu"],
        "comet": metrics["comet"]
    })
    print(metrics)
    
    del model
    del inputs
    del preds
    del labels
    
    torch.cuda.empty_cache()
    
    gc.collect()

df_results = pd.DataFrame(results_list)
df_results.to_csv("cascade_finetuned.csv", index=False)
print("Results saved to cascade_finetuned.csv")


