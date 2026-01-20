# %% [markdown]
# # A2 Coursework Neural Models
# ## NLLB models

# %% [markdown]
# ### Imports

# %%
import torch
import numpy as np

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
rouge_metric = load("rouge")
chrf_metric = load("chrf")

DEBUG_MODE = False
DEBUG_FRACTION = 0.05

# %%
opus_dataset = load_dataset("Helsinki-NLP/opus-100", "en-fr")

if DEBUG_MODE:
    opus_dataset = DatasetDict({
        split: opus_dataset[split]
            .shuffle(seed=42)
            .select(range(int(len(opus_dataset[split]) * DEBUG_FRACTION)))
        for split in opus_dataset.keys()
    })

print(opus_dataset)

# %%
max_tok_length = 16
src_code_nllb = "eng_Latn"
tgt_code_nllb = "fra_Latn"

src_code_mbart = "en_XX"
tgt_code_mbart = "fr_XX"

checkpoint_nllb = "facebook/nllb-200-distilled-600M"
checkpoint_mbart = "facebook/mbart-large-50-many-to-many-mmt"

tokenizer_nllb = AutoTokenizer.from_pretrained(
    checkpoint_nllb, 
    padding=True, 
    pad_to_multiple_of=8, 
    src_lang=src_code_nllb, 
    tgt_lang=tgt_code_nllb, 
    truncation=True, 
    max_length=max_tok_length,
)

tokenizer_mbart = AutoTokenizer.from_pretrained(
    checkpoint_mbart,
    padding=True,
    pad_to_multiple_of=8,
    src_lang=src_code_mbart,
    tgt_lang=tgt_code_mbart,
    truncation=True,
    max_length=max_tok_length,
)

# %%
source_lang = "en"
target_lang = "fr"

def preprocess_function_opus(batch, tokenizer):
    source_texts = [t[source_lang] for t in batch["translation"]]
    target_texts = [t[target_lang] for t in batch["translation"]]
    
    model_inputs = tokenizer(
        source_texts,
        text_target=target_texts,
        truncation=True,
        max_length=max_tok_length
    )
    
    return model_inputs


# %%
tokenized_datasets_nllb = opus_dataset.map(
    lambda batch: preprocess_function_opus(batch, tokenizer_nllb),
    batched=True, 
    num_proc=8
)

tokenized_datasets_nllb = tokenized_datasets_nllb.filter(
    lambda x: len(x["input_ids"]) <= max_tok_length and len(x["labels"]) <= max_tok_length,
    desc=f"Discarding source and target sentences with more than {max_tok_length} tokens", num_proc=8
)

tokenized_datasets_mbart = opus_dataset.map(
    lambda batch: preprocess_function_opus(batch, tokenizer_mbart),
    batched=True, 
    num_proc=8
)

tokenized_datasets_mbart = tokenized_datasets_mbart.filter(
    lambda x: len(x["input_ids"]) <= max_tok_length and len(x["labels"]) <= max_tok_length,
    desc=f"Discarding source and target sentences with more than {max_tok_length} tokens", num_proc=8
)

# %%
def show_length_distribution(tokenized_datasets):
    dic = {}
    for sample in tokenized_datasets['train']:
        sample_length = len(sample['input_ids'])
        if sample_length not in dic:
            dic[sample_length] = 1
        else:
            dic[sample_length] += 1 

    for i in range(1,max_tok_length+1):
        if i in dic:
            print(f"{i:>2} {dic[i]:>3}")
            
show_length_distribution(tokenized_datasets_nllb)
show_length_distribution(tokenized_datasets_mbart)

# %%
def build_seq2seq_model(
    checkpoint,
    quantization_config,
    tokenizer = None,
    peft_config = None,
    train_mode = False
):
    model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint,
        quantization_config=quantization_config
    )

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    if not train_mode:
        return model

    model = get_peft_model(model, peft_config)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=8
    )

    return model, collator


# %%
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# %%
LoraConfig_nllb = LoraConfig(
    task_type="SEQ_2_SEQ_LM",
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

LoraConfig_mbart = LoraConfig(
    task_type="SEQ_2_SEQ_LM",
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

IA3Config_mbart = IA3Config(
    task_type="SEQ_2_SEQ_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    feedforward_modules=["fc1", "fc2"]
)

# %%
batch_size = 32
    
def setup_training_args(model_name, lr = None, extra = None):
    output_save_dir = model_name
    if not lr is None:
        output_save_dir += f"-lr{lr}"
    if not extra is None:
        output_save_dir += f"-{extra}"
        
    return Seq2SeqTrainingArguments(
        output_dir=output_save_dir,
        eval_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=2,
        predict_with_generate=True,
        logging_strategy="epoch",
        disable_tqdm=True
    )


# %%
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    
    return preds, labels

def compute_metrics(eval_preds, tokenizer, training = False):
    if training:
        preds, labels = eval_preds
    else:
        inputs, preds, labels = eval_preds

    if not training and not isinstance(inputs, list):
        inputs = list(inputs)
    if not isinstance(labels, list):
        labels = list(labels)
        
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace negative ids in labels as we can't decode them.
    if not training:    
        inputs = [
            [tokenizer.pad_token_id if j < 0 else j for j in input]
            for input in inputs
        ]
        decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)

    labels = [
        [tokenizer.pad_token_id if j < 0 else j for j in label]
        for label in labels
    ]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    #BLEU
    bleu_result = bleu_metric.compute(predictions=decoded_preds, references=[decoded_labels[i] for i in range(len(decoded_labels))])
    result = {"bleu": bleu_result["score"]}
    
    # ROUGE
    rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    result["rougeL"] = rouge_result["rougeL"]

    # chrF
    chrf_result = chrf_metric.compute(predictions=decoded_preds, references=decoded_labels)
    result["chrf"] = chrf_result["score"]
    
    # COMET
    if not training:
        comet_result = comet_metric.compute(sources=decoded_inputs, predictions=decoded_preds, references=decoded_labels)
        result["comet"] = comet_result["mean_score"] * 100


    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# %%
def build_trainer(
    model,
    collator,
    tokenizer,
    train_dataset,
    eval_dataset,
    training_args,
):
    return Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=lambda eval_preds: compute_metrics(
            eval_preds, tokenizer, training=True
        )
    )


# %%
# model_baseline_nllb = build_seq2seq_model(
#     checkpoint_nllb,
#     quantization_config,
#     train_mode=False
# )

# # LR = 5e-5
# model_nllb_lr5e5, collator_nllb_lr5e5 = build_seq2seq_model(
#     checkpoint_nllb,
#     quantization_config,
#     tokenizer_nllb,
#     LoraConfig_nllb,
#     train_mode=True
# )

# trainer_nllb_lr5e5 = build_trainer(
#     model_nllb_lr5e5,
#     collator_nllb_lr5e5,
#     tokenizer_nllb,
#     tokenized_datasets_nllb["train"],
#     tokenized_datasets_nllb["validation"],
#     setup_training_args("nllb", lr = 5e-5)
# )

# trainer_nllb_lr5e5.train()


# # LR = 1e-4
# model_nllb_lr1e4, collator_nllb_lr1e4 = build_seq2seq_model(
#     checkpoint_nllb,
#     quantization_config,
#     tokenizer_nllb,
#     LoraConfig_nllb,
#     train_mode=True
# )

# trainer_nllb_lr1e4 = build_trainer(
#     model_nllb_lr1e4,
#     collator_nllb_lr1e4,
#     tokenizer_nllb,
#     tokenized_datasets_nllb["train"],
#     tokenized_datasets_nllb["validation"],
#     setup_training_args("nllb", lr = 1e-4)
# )

# trainer_nllb_lr1e4.train()


# %%
model_baseline_mbart = build_seq2seq_model(
    checkpoint_mbart,
    quantization_config,
    train_mode=False
)

model_finetuned_mbart_lora, collator_mbart = build_seq2seq_model(
    checkpoint_mbart,
    quantization_config,
    tokenizer_mbart,
    LoraConfig_mbart,
    train_mode=True
)

trainer_mbart_lora = build_trainer(
    model_finetuned_mbart_lora,
    collator_mbart,
    tokenizer_mbart,
    tokenized_datasets_mbart["train"],
    tokenized_datasets_mbart["validation"],
    setup_training_args("mbart", lr = 1e-4, extra="lora")
)

trainer_mbart_lora.train()

model_finetuned_mbart_ia3, collator_mbart = build_seq2seq_model(
    checkpoint_mbart,
    quantization_config,
    tokenizer_mbart,
    IA3Config_mbart,
    train_mode=True
)

trainer_mbart_ia3 = build_trainer(
    model_finetuned_mbart_ia3,
    collator_mbart,
    tokenizer_mbart,
    tokenized_datasets_mbart["train"],
    tokenized_datasets_mbart["validation"],
    setup_training_args("mbart", lr = 1e-4, extra="ia3")
)

trainer_mbart_ia3.train()

# %% [markdown]
# ## Inference

# %%
generation_config_nllb = GenerationConfig.from_pretrained(
    checkpoint_nllb,
)

generation_config_mbart = GenerationConfig.from_pretrained(
    checkpoint_mbart,
)

# %%
batch_tokenized_test_nllb = tokenized_datasets_nllb['test'].batch(batch_size)
batch_tokenized_test_mbart = tokenized_datasets_mbart['test'].batch(batch_size)

# %%
def evaluate_model(model, batch_tokenized_test, tokenizer, generation_config, n_beams=1):
    number_of_batches = len(batch_tokenized_test["translation"])
    input_sequences = []
    preds_sequences = []
    labels_sequences = []
    for i in range(number_of_batches):
        batch_tokenized_test_src = list(batch_tokenized_test["translation"][i][j][source_lang] for j in range(len(batch_tokenized_test["translation"][i])))
        batch_tokenized_test_tgt = list(batch_tokenized_test["translation"][i][j][target_lang] for j in range(len(batch_tokenized_test["translation"][i])))
        inputs = tokenizer(
            batch_tokenized_test_src, 
            max_length=max_tok_length, 
            truncation=True, 
            return_tensors="pt", 
            padding=True,
            )
        labels = tokenizer(
            batch_tokenized_test_tgt, 
            max_length=max_tok_length, 
            truncation=True, 
            return_tensors="pt", 
            padding=True,
        )
        with torch.no_grad():    
            
            bos_id = None
            
            if tokenizer == tokenizer_nllb:
                bos_id = tokenizer.convert_tokens_to_ids(tgt_code_nllb)
            else:
                bos_id = tokenizer.lang_code_to_id[tgt_code_mbart]
            
            output_batch = model.generate(
                generation_config=generation_config, 
                input_ids=inputs["input_ids"].cuda(), 
                attention_mask=inputs["attention_mask"].cuda(), 
                forced_bos_token_id=bos_id, 
                max_length = max_tok_length, 
                num_beams=n_beams, 
                do_sample=False,
                )
        input_sequences.extend(inputs["input_ids"].cpu())
        preds_sequences.extend(output_batch.cpu())
        labels_sequences.extend(labels["input_ids"].cpu())
    return input_sequences, preds_sequences, labels_sequences

# %% [markdown]
# ## Results

# %%
import pandas as pd

# Definiamo le configurazioni da testare
eval_configs = [
    # {"model": model_baseline_nllb,          "tokenizer": tokenizer_nllb, "name": "NLLB", "finetuned": False, "lr": None, "num_beams": 1},
    # {"model": model_nllb_lr5e5,             "tokenizer": tokenizer_nllb, "name": "NLLB", "finetuned": True,  "lr": 5e-5, "num_beams": 1},
    # {"model": model_nllb_lr1e4,             "tokenizer": tokenizer_nllb, "name": "NLLB", "finetuned": True,  "lr": 1e-4, "num_beams": 1},
    # {"model": model_baseline_nllb,          "tokenizer": tokenizer_nllb, "name": "NLLB", "finetuned": False, "lr": None, "num_beams": 4},
    # {"model": model_nllb_lr5e5,             "tokenizer": tokenizer_nllb, "name": "NLLB", "finetuned": True,  "lr": 5e-5, "num_beams": 4},
    # {"model": model_nllb_lr1e4,             "tokenizer": tokenizer_nllb, "name": "NLLB", "finetuned": True,  "lr": 1e-4, "num_beams": 4},
    {"model": model_baseline_mbart,         "tokenizer": tokenizer_mbart,"name": "MBART","finetuned": False, "peft": "Lora", "num_beams": 1},
    {"model": model_finetuned_mbart_lora,   "tokenizer": tokenizer_mbart,"name": "MBART","finetuned": True,  "peft": "Lora", "num_beams": 1},
    {"model": model_finetuned_mbart_ia3, "tokenizer": tokenizer_mbart,"name": "MBART","finetuned": True,  "peft": "IA3", "num_beams": 1},
    {"model": model_baseline_mbart,         "tokenizer": tokenizer_mbart,"name": "MBART","finetuned": False, "peft": "Lora", "num_beams": 4},
    {"model": model_finetuned_mbart_lora,   "tokenizer": tokenizer_mbart,"name": "MBART","finetuned": True,  "peft": "Lora", "num_beams": 4},
    {"model": model_finetuned_mbart_ia3, "tokenizer": tokenizer_mbart,"name": "MBART","finetuned": True,  "peft": "IA3", "num_beams": 4},
]

import pandas as pd

results = []

for cfg in eval_configs:
    model = cfg["model"]
    tokenizer = cfg["tokenizer"]
    name = cfg["name"]
    finetuned = cfg["finetuned"]
    lr = cfg.get("lr", None)
    peft = cfg.get("peft", None)
    num_beams = cfg["num_beams"]

    batch_tokenized_test = batch_tokenized_test_nllb if name == "NLLB" else batch_tokenized_test_mbart

    input_seqs, pred_seqs, label_seqs = evaluate_model(
        model, batch_tokenized_test, tokenizer, 
        generation_config_nllb if name=="NLLB" else generation_config_mbart,
        n_beams=num_beams
    )

    metrics = compute_metrics((input_seqs, pred_seqs, label_seqs), tokenizer)

    result = {
        "Model": name,
        "Finetuned": finetuned,
        "Decoding": "Greedy" if num_beams==1 else f"Beam {num_beams}",
        "BLEU": metrics["bleu"],
        "COMET": metrics["comet"],
        "ROUGE-L": metrics["rougeL"],
        "chrF": metrics["chrf"],
    }

    if lr is not None:
        result["LR"] = lr
    if peft is not None:
        result["PEFT"] = peft

    results.append(result)

df_results = pd.DataFrame(results)

cols_order = ["Model", "Finetuned", "LR", "PEFT", "Decoding", "BLEU", "COMET", "ROUGE-L", "chrF"]
df_results = df_results.reindex(columns=[c for c in cols_order if c in df_results.columns])


df_results["ΔBLEU"] = df_results.groupby(["Model", "Decoding"])["BLEU"].transform(lambda x: x - x.iloc[0])
df_results["ΔROUGE-L"] = df_results.groupby(["Model", "Decoding"])["ROUGE-L"].transform(lambda x: x - x.iloc[0])
df_results["ΔchrF"] = df_results.groupby(["Model", "Decoding"])["chrF"].transform(lambda x: x - x.iloc[0])
df_results["ΔCOMET"] = df_results.groupby(["Model", "Decoding"])["COMET"].transform(lambda x: x - x.iloc[0])

df_results = df_results.sort_values(by=["Model", "Decoding", "Finetuned"], ascending=[True, True, False])

df_nllb = df_results[df_results["Model"]=="NLLB"].copy()
df_mbart = df_results[df_results["Model"]=="MBART"].copy()

# print(df_nllb.to_string(index=False))
# df_nllb.to_csv("results_nllb.csv", index=False)

print(df_mbart.to_string(index=False))
df_mbart.to_csv("results_mbart.csv", index=False)



