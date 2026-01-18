# %% [markdown]
# # A2 Coursework Neural Models

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
from transformers import Trainer, Seq2SeqTrainer
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Seq2SeqTrainingArguments 
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from evaluate import load


login(token="")

bleu_metric = load("sacrebleu")
comet_metric = load("comet")

DEBUG_MODE = True
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
src_code = "eng_Latn"
tgt_code = "fra_Latn"

checkpoint_nllb = "facebook/nllb-200-distilled-600M"
checkpoint_llama = "meta-llama/Llama-2-7b-hf"

tokenizer_nllb = AutoTokenizer.from_pretrained(
    checkpoint_nllb, 
    padding=True, 
    pad_to_multiple_of=8, 
    src_lang=src_code, 
    tgt_lang=tgt_code, 
    truncation=True, 
    max_length=max_tok_length,
)

tokenizer_llama = AutoTokenizer.from_pretrained(
    checkpoint_llama, 
    token=True,
    padding=True,
    pad_to_multiple_of=8,
    truncation=True,
    max_length=max_tok_length,
    padding_side='left',
    )
tokenizer_llama.pad_token = tokenizer_llama.eos_token

# %%
source_lang = "en"
target_lang = "fr"

def preprocess_function_opus(batch, tokenizer, model_name = "nllb"):
    source_texts = [t[source_lang] for t in batch["translation"]]
    target_texts = [t[target_lang] for t in batch["translation"]]
    
    args = {
        "truncation": True,
        "max_length": max_tok_length,
    } if model_name == "nllb" else {}
    
    model_inputs = tokenizer(
        source_texts,
        text_target=target_texts,
        **args
    )
    
    return model_inputs


# %%
tokenized_datasets_nllb = opus_dataset.map(
    lambda batch: preprocess_function_opus(batch, tokenizer_nllb, model_name="nllb"),
    batched=True, 
    num_proc=8
)

tokenized_datasets_nllb = tokenized_datasets_nllb.filter(
    lambda x: len(x["input_ids"]) <= max_tok_length and len(x["labels"]) <= max_tok_length,
    desc=f"Discarding source and target sentences with more than {max_tok_length} tokens", num_proc=8
)

tokenized_datasets_llama = opus_dataset.map(
    lambda batch: preprocess_function_opus(batch, tokenizer_llama, model_name="llama"),
    batched=True, 
    num_proc=8
)

tokenized_datasets_llama = tokenized_datasets_llama.filter(
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
show_length_distribution(tokenized_datasets_llama)

# %%
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# %%
model_nllb_baseline = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint_nllb,
    quantization_config=quantization_config
)

model_nllb_finetuned = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint_nllb,
    quantization_config=quantization_config
)

model_llama_promting = AutoModelForCausalLM.from_pretrained(
    checkpoint_llama,
    token=True,
    quantization_config=quantization_config,
    dtype=torch.bfloat16,
)

model_llama_finetuned = AutoModelForCausalLM.from_pretrained(
    checkpoint_llama,
    token=True,
    quantization_config=quantization_config,
    dtype=torch.bfloat16,
)

# %%
model_nllb_finetuned = prepare_model_for_kbit_training(
    model_nllb_finetuned,
    use_gradient_checkpointing=False,
    gradient_checkpointing_kwargs={'use_reentrant':False}
)

model_llama_finetuned = prepare_model_for_kbit_training(
    model_llama_finetuned,
    use_gradient_checkpointing=False,
    gradient_checkpointing_kwargs={'use_reentrant':False}
)

# %% [markdown]
# ## Configuration

# %%
LoraConfig_nllb = LoraConfig(
    task_type="SEQ_2_SEQ_LM",
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)

# %%
LoraConfig_llama = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    inference_mode=False,
)

# %%
model_nllb_finetuned = get_peft_model(model_nllb_finetuned, LoraConfig_nllb)
print(model_nllb_finetuned.print_trainable_parameters())

model_llama_finetuned = get_peft_model(model_llama_finetuned, LoraConfig_llama)
print(model_llama_finetuned.print_trainable_parameters())

# %%
data_collator_nllb = DataCollatorForSeq2Seq(
    tokenizer = tokenizer_nllb,
    model = model_nllb_finetuned,
    pad_to_multiple_of=8
)

data_collator_llama = DataCollatorForLanguageModeling(
    tokenizer=tokenizer_llama, 
    mlm=False, 
    pad_to_multiple_of=8
)

# %%
batch_size_llama = 4
gradient_accumulation_steps = 8
model_name_llama = checkpoint_llama.split("/")[-1]
args_llama = TrainingArguments(
    f"{model_name_llama}-finetuned-en-to-fr",
    eval_strategy = "epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=batch_size_llama,
    per_device_eval_batch_size=batch_size_llama,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    warmup_steps=100,
    optim="adamw_bnb_8bit",
    prediction_loss_only=True,
    gradient_accumulation_steps = gradient_accumulation_steps,
    bf16=True,
    bf16_full_eval=True,
    group_by_length=True,
)

batch_size_nllb = 32
model_name_nllb = checkpoint_nllb.split("/")[-1]
args_nllb = Seq2SeqTrainingArguments(
    f"{model_name_nllb}-finetuned-en-to-fr",
    eval_strategy = "epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=batch_size_nllb,
    per_device_eval_batch_size=batch_size_nllb,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=2,
    predict_with_generate=True,
    logging_strategy="epoch",
    disable_tqdm=True
)

# %% [markdown]
# #### Llama preprocessing

# %%
import torch
src = "en"
tgt = "fr"
task_prefix = f"Translate from {src} to {tgt}:\n"
s = ""

prefix_tok_len = len(tokenizer_llama.encode(f"{task_prefix}{src}: {s} = {tgt}: "))
max_tok_len_training = prefix_tok_len
# Adding 2 for new line in target sentence and eos_token_id token
max_tok_len_training += 2 * max_tok_length + 2

def preprocess4training_function(sample, max_tok_len):
    
    sample_size = len(sample["translation"])

    # Creating the prompt with the task description for each source sentence
    inputs  = [f"{task_prefix}{src}: {s[src]} = {tgt}: " for s in sample["translation"]]

    # Appending new line after each sample in the batch
    targets = [f"{s[tgt]}\n" for s in sample["translation"]]

    # Applying the Llama2 tokenizer to the inputs and targets 
    # to obtain "input_ids" (token_ids) and "attention mask" 
    model_inputs = tokenizer_llama(inputs)
    labels = tokenizer_llama(targets)
    
    # Each input is appended with its target 
    # Each target is prepended with as many special token id (-100) as the original input length
    # Both input and target (label) has the same max_tok_len
    # Attention mask is all 1s 
    for i in range(sample_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer_llama.eos_token_id]
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

    # Each input is applied left padding up to max_tok_len
    # Attention mask is 0 for padding
    # Each target (label) is left filled with special token id (-100)
    # Finally inputs, attention_mask and targets (labels) are truncated to max_tok_len
    for i in range(sample_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer_llama.pad_token_id] * (
            max_tok_len - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_tok_len - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (max_tok_len - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_tok_len])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_tok_len])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_tok_len])
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# %%
num_shots = 1
shots = ""

shot_tok_len   = len(tokenizer_llama.encode(f"{src}: {s} = {tgt}: {s}\n"))
max_tok_len_test = prefix_tok_len
max_tok_len_test += num_shots * (shot_tok_len + 2 * max_tok_length) 
max_tok_len_test += max_tok_length

def preprocess4test_function(sample):
    inputs = [f"{task_prefix}{shots}{src}: {s} = {tgt}: " for s in sample["source_text"]]
    model_inputs = tokenizer_llama(
        inputs,
        max_length=max_tok_len_test, 
        truncation=True, 
        return_tensors="pt", 
        padding=True)
    return model_inputs

# %%
preprocessed_train_dataset = tokenized_datasets_llama['train'].map(preprocess4training_function, batched=True)
preprocessed_dev_dataset = tokenized_datasets_llama['validation'].map(preprocess4training_function, batched=True)

# %%
preprocessed_test_dataset = tokenized_datasets_llama['test'].map(preprocess4test_function, batched=True)

# %%
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    
    return preds, labels

def compute_metrics(eval_preds, tokenizer):
    training = True if len(eval_preds) == 2 else False
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
    

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    bleu_result = bleu_metric.compute(predictions=decoded_preds, references=[decoded_labels[i] for i in range(len(decoded_labels))])
    result = {"bleu": bleu_result["score"]}
    
    if not training:
        comet_result = comet_metric.compute(sources=decoded_inputs, predictions=decoded_preds, references=decoded_labels)
        result["comet"] = comet_result["mean_score"] * 100


    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# %%
trainer_llama = Trainer(
    model_llama_finetuned,
    args_llama,
    train_dataset=preprocessed_train_dataset,
    eval_dataset=preprocessed_dev_dataset,
    processing_class=tokenizer_llama,
    data_collator=data_collator_llama,
)

trainer_nllb = Seq2SeqTrainer(
    model_nllb_finetuned,
    args_nllb,
    train_dataset=tokenized_datasets_nllb['train'],
    eval_dataset=tokenized_datasets_nllb['validation'],
    processing_class=tokenizer_nllb,
    data_collator=data_collator_nllb,
    compute_metrics = lambda eval_preds: compute_metrics(eval_preds, tokenizer_nllb)
)

# %%
print(" ---- NLLB Training ----")
trainer_nllb.train()

# %%
print(" ---- LLaMA Training ----")
trainer_llama.train()

# %% [markdown]
# ## Inference

# %%
generation_config_nllb = GenerationConfig.from_pretrained(
    checkpoint_nllb,
)
    
generation_config_llama = GenerationConfig.from_pretrained(
    checkpoint_llama,
)

# %%
batch_tokenized_test_nllb = tokenized_datasets_nllb['test'].batch(batch_size_nllb)
batch_tokenized_test_llama = preprocessed_test_dataset.batch(batch_size_llama)

# %%
def evaluate_model(model, batch_tokenized_test, tokenizer, generation_config):
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
            output_batch = model.generate(
                generation_config=generation_config, 
                input_ids=inputs["input_ids"].cuda(), 
                attention_mask=inputs["attention_mask"].cuda(), 
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_code), 
                max_length = max_tok_length, 
                num_beams=1, 
                do_sample=False,
                )
        input_sequences.extend(inputs["input_ids"].cpu())
        preds_sequences.extend(output_batch.cpu())
        labels_sequences.extend(labels["input_ids"].cpu())
    return input_sequences, preds_sequences, labels_sequences

# %%
def evaluate_model(
    model,
    batch_tokenized_test,
    tokenizer,
    generation_config,
    source_lang,
    target_lang,
    max_tok_length,
    tgt_code=None,
    device="cuda"
):
    model.eval()

    number_of_batches = len(batch_tokenized_test["translation"])
    input_sequences = []
    preds_sequences = []
    labels_sequences = []

    for i in range(number_of_batches):
        batch_src = [
            batch_tokenized_test["translation"][i][j][source_lang]
            for j in range(len(batch_tokenized_test["translation"][i]))
        ]
        batch_tgt = [
            batch_tokenized_test["translation"][i][j][target_lang]
            for j in range(len(batch_tokenized_test["translation"][i]))
        ]

        inputs = tokenizer(
            batch_src,
            max_length=max_tok_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        labels = tokenizer(
            batch_tgt,
            max_length=max_tok_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        gen_kwargs = dict(
            generation_config=generation_config,
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            max_length=max_tok_length,
            num_beams=1,
            do_sample=False,
        )

        if tgt_code is not None:
            gen_kwargs["forced_bos_token_id"] = tokenizer.convert_tokens_to_ids(tgt_code)

        with torch.no_grad():
            outputs = model.generate(**gen_kwargs)

        input_sequences.extend(inputs["input_ids"].cpu())
        preds_sequences.extend(outputs.cpu())
        labels_sequences.extend(labels["input_ids"].cpu())

    return input_sequences, preds_sequences, labels_sequences


# %% [markdown]
# ## Results

# %%
print(" ---- NLLB Results ----")

input_sequences, preds_sequences, labels_sequences = evaluate_model(model_nllb_baseline, batch_tokenized_test_nllb, tokenizer_nllb, generation_config_nllb, source_lang, target_lang, max_tok_length, tgt_code)
print("Baseline:")
nllb_baseline_result = compute_metrics((input_sequences, preds_sequences, labels_sequences), tokenizer_nllb)
print(f'BLEU: {nllb_baseline_result["bleu"]}')
print(f'COMET: {nllb_baseline_result["comet"]}')

input_sequences, preds_sequences, labels_sequences = evaluate_model(model_nllb_finetuned, batch_tokenized_test_nllb, tokenizer_nllb, generation_config_nllb, source_lang, target_lang, max_tok_length, tgt_code)
print("Finetuned:")
nllb_finetuned_result = compute_metrics((input_sequences, preds_sequences, labels_sequences), tokenizer_nllb)
print(f'BLEU: {nllb_finetuned_result["bleu"]}')
print(f'COMET: {nllb_finetuned_result["comet"]}')

# %%
print(" ---- LLaMA Results ----")

input_sequences, preds_sequences, labels_sequences = evaluate_model(model_llama_promting, batch_tokenized_test_llama, tokenizer_llama, generation_config_llama, source_lang, target_lang, max_tok_len_test)
print("Prometing:")
llama_promting_result = compute_metrics((input_sequences, preds_sequences, labels_sequences), tokenizer_llama)
print(f'BLEU: {llama_promting_result["bleu"]}')
print(f'COMET: {llama_promting_result["comet"]}')

input_sequences, preds_sequences, labels_sequences = evaluate_model(model_llama_finetuned, batch_tokenized_test_llama, tokenizer_llama, generation_config_llama, source_lang, target_lang, max_tok_len_test)
print("Finetuned:")
llama_finetuned_result = compute_metrics((input_sequences, preds_sequences, labels_sequences), tokenizer_llama)
print(f'BLEU: {llama_finetuned_result["bleu"]}')
print(f'COMET: {llama_finetuned_result["comet"]}')


