# %% [markdown]
# # A2 Coursework Neural Models
# ## OPT models prompting

# %% [markdown]
# ### Imports

# %%
import torch
import numpy as np
import pandas as pd
import gc
import time

from datasets import load_dataset, DatasetDict
from huggingface_hub import login
from transformers import AutoTokenizer
from transformers import GenerationConfig
from transformers import BitsAndBytesConfig
from transformers import Trainer
from transformers import AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments 
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from peft import IA3Config, PrefixTuningConfig
from evaluate import load

torch.set_float32_matmul_precision('medium')

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
max_tok_length = 24

checkpoint_opt = "facebook/opt-1.3b"

tokenizer_opt = AutoTokenizer.from_pretrained(
    checkpoint_opt, 
    padding=True,
    pad_to_multiple_of=8,
    truncation=True,
    max_length=max_tok_length,
    padding_side='left',
)

special_tokens = {
    "additional_special_tokens": ["<END>"]
}

tokenizer_opt.add_special_tokens(special_tokens)

tokenizer_opt.pad_token = tokenizer_opt.eos_token

# %%
src_lang = "en"
tag_lang = "fr"

source_language = "English"
target_language = "French"

task_prefix = f"Translate from {source_language} to {target_language}:\n"    

def preprocess_function_opus(batch, tokenizer):
    source_texts = [t[src_lang] for t in batch["translation"]]
    target_texts = [t[tag_lang] for t in batch["translation"]]
    
    model_inputs = tokenizer(
        source_texts,
        text_target=target_texts,
    )
    
    return model_inputs


# %%
opus_dataset_opt = opus_dataset.map(
    lambda batch: preprocess_function_opus(batch, tokenizer_opt),
    batched=True, 
    num_proc=8
)

opus_dataset_opt = opus_dataset_opt.filter(
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
            
show_length_distribution(opus_dataset_opt)

# %%
def get_training_max_tok_len(task_prefix, tokenizer):
    s = ""
    prefix_tok_len = len(tokenizer.encode(f"{task_prefix}{src_lang}: {s} = {tag_lang}: "))
    max_tok_len = prefix_tok_len
    # Adding 2 for new line in target sentence and eos_token_id token
    max_tok_len += 2 * max_tok_length + 2
    return max_tok_len

def preprocess4training_function(sample, task_prefix, tokenizer):

    max_tok_len = get_training_max_tok_len(task_prefix, tokenizer)
    sample_size = len(sample["translation"])
    inputs  = [f"{task_prefix}{source_language}: {s[src_lang]} = {target_language}: " for s in sample["translation"]]

    # targets = [f"{s[tag_lang]}\n" for s in sample["translation"]]
    targets = [f"{s[tag_lang]} <END>" for s in sample["translation"]]

    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)
    
    for i in range(sample_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])

    for i in range(sample_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
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
def get_test_max_tok_len(num_shots, task_prefix, tokenizer):
    
    s = ""
    shots = ""
    prefix_tok_len = len(tokenizer.encode(f"{task_prefix}{shots}{src_lang}: {s} = {tag_lang}: "))
    shot_tok_len   = len(tokenizer.encode(f"{src_lang}: {s} = {tag_lang}: {s}\n"))
    max_tok_len = prefix_tok_len
    max_tok_len += num_shots * (shot_tok_len + 2 * max_tok_length) 
    max_tok_len += max_tok_length
    return task_prefix, max_tok_len 

def preprocess4test_function(test_sample, task_prefix, tokenizer, training_sample=None, num_shots = 1):
        
    if training_sample is None:
        inputs = [f"{task_prefix}{source_language}: {s[src_lang]} = {target_language}: " for s in test_sample["translation"]]
        model_inputs = tokenizer(inputs,padding=True,)
        return model_inputs
    
    task_prefix, max_tok_len = get_test_max_tok_len(num_shots, task_prefix, tokenizer)
    
    shots = ""

    random_seed = time.time()
    t_sample = training_sample.shuffle(seed=int(random_seed)).select(range(num_shots))
    for s in t_sample["translation"]: shots += f"{source_language}: {s[src_lang]} = {target_language}: {s[tag_lang]} <END>"
    
    inputs = [f"{task_prefix}{shots}{source_language}: {s[src_lang]} = {target_language}: " for s in test_sample["translation"]]
    model_inputs = tokenizer(
        inputs,
        max_length=max_tok_len, 
        truncation=True, 
        return_tensors="pt", 
        padding=True)
    return model_inputs

# %%
trainset_opt = opus_dataset_opt['train'].map(
    lambda x: preprocess4training_function(x, task_prefix, tokenizer_opt),
    batched=True
)

devset_opt = opus_dataset_opt['validation'].map(
    lambda x: preprocess4training_function(x, task_prefix, tokenizer_opt),
    batched=True
)

testset_opt_trained = opus_dataset_opt['test'].map(
    lambda x: preprocess4test_function(x, task_prefix, tokenizer_opt, num_shots=0),
    batched=True
)

# %%
for sample in testset_opt_trained.select(range(5)):
    print(sample['input_ids'])
    print(sample['attention_mask'])
    print(sample['labels'])

# %%
def build_seq2seq_model(
    checkpoint,
    quantization_config,
    tokenizer = None,
    peft_config = None,
    train_mode = False
):
    
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        token=True,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map="auto",
        # attn_implementation="flash_attention_2"
    )
    
    model.resize_token_embeddings(len(tokenizer))

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=False,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    if not train_mode:
        return model

    model = get_peft_model(model, peft_config)
    print(f"Model {checkpoint} Trainable Parameters:")
    model.print_trainable_parameters()
    
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False, 
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
LoraConfig_opt = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "v_proj"],
)

IA3Config_opt = IA3Config(
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "fc1",
        "fc2",
    ],
    feedforward_modules=[
        "fc1",
        "fc2",
    ],
)


# %%
batch_size = 32
gradient_accumulation_steps = 1

def setup_training_args(model_name, lr):
    return TrainingArguments(
        output_dir=f"{model_name}-lr{lr}",
        eval_strategy = "epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=2,
        warmup_steps=100,
        optim="adamw_bnb_8bit",
        prediction_loss_only=True,
        gradient_accumulation_steps = gradient_accumulation_steps,
        bf16=True,
        bf16_full_eval=True,
        group_by_length=True,
        disable_tqdm=True,
    )

# %%
def build_trainer(
    model,
    collator,
    tokenizer,
    train_dataset,
    eval_dataset,
    training_args,
):
    
    return Trainer(
        model = model,
        args = training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
    )


# %%
generation_config_opt = GenerationConfig.from_pretrained(
    checkpoint_opt,
)

# %%
print("Generation Config OPT:", generation_config_opt)

# %%
test_batch_size = 32
testset_opt_trained_batched = testset_opt_trained.batch(test_batch_size)

# %%
def evaluate_model(
    model,
    batch_tokenized_test,
    tokenizer,
    generation_config,
    max_tok_len,
    num_beams=1,
):
    model.eval()
    input_sequences = []
    preds_sequences = []
    labels_sequences = []
    
    end_token_id = tokenizer.convert_tokens_to_ids("<END>")
    
    for batch in batch_tokenized_test:
        input_ids = torch.tensor(batch["input_ids"]).cuda()
        attention_mask = torch.tensor(batch["attention_mask"]).cuda()
            
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                max_new_tokens= max_tok_length,
                # max_length= max_tok_len,
                num_beams=num_beams,
                do_sample=False,
                eos_token_id=end_token_id,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
        
        preds_sequences.extend(
            tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
        )
        
        input_sequences.extend(
            tokenizer.batch_decode(
                input_ids, skip_special_tokens=True
            )
        )      
        
        labels_sequences.extend(
            tokenizer.batch_decode(
                batch["labels"], skip_special_tokens=True
            )
        )
    return input_sequences, preds_sequences, labels_sequences
    

# %%
def compute_metrics(sample, tokenizer):
    inputs, preds, labels = sample

    clean_preds = []

    for input, pred in zip(inputs, preds):
        text = pred.removeprefix(input).strip()

        if "<END>" in text:
            clean_preds.append(text.split("<END>")[0].strip())
        else:
            clean_preds.append(text)

    results = {}
    results["BLEU"] = bleu_metric.compute(
        predictions=clean_preds,
        references=[[s] for s in labels]
    )["score"]
    
    results["rogueL"] = rouge_metric.compute(
        predictions=clean_preds,
        references=labels
    )["rougeL"]
    
    results["COMET"] = comet_metric.compute(
        predictions=clean_preds,
        references=labels,
        sources = inputs
    )["mean_score"] * 100
    
    results["chrF"] = chrf_metric.compute(
        predictions=clean_preds,
        references=labels
    )["score"]
    
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    results["gen_len"] = np.mean(prediction_lens)
    results = {k: round(v, 4) for k, v in results.items()}
    return results

# %%
cfg_list = []

for model_name, base_cfg in {
    "opt_lora_5e5": (checkpoint_opt, LoraConfig_opt, 5e-5, tokenizer_opt),
    "opt_lora_1e4": (checkpoint_opt, LoraConfig_opt, 1e-4, tokenizer_opt),
    "opt_ia3_5e5": (checkpoint_opt, IA3Config_opt, 5e-5, tokenizer_opt),
    "opt_ia3_1e4": (checkpoint_opt, IA3Config_opt, 1e-4, tokenizer_opt),
}.items():
    checkpoint, config, lr, tokenizer = base_cfg
    cfg_list.append({
        "model_name": model_name,
        "checkpoint": checkpoint,
        "peft_config": config,
        "lr": lr,
        "tokenizer": tokenizer,
    })
    
results_list = []

for cfg_item in cfg_list:
    name = cfg_item["model_name"]
    
    model, collator = build_seq2seq_model(
        checkpoint_opt,
        quantization_config,
        cfg_item["tokenizer"],
        cfg_item["peft_config"],
        train_mode=True
    )

    trainer = build_trainer(
        model,
        collator,
        cfg_item["tokenizer"],
        trainset_opt,
        devset_opt,
        setup_training_args("opt", lr)
    )

    trainer.train()
    
    for n_b in [1, 4]:
        num_beams = n_b
        lr = cfg_item["lr"]
        print(f"Evaluating model: {name} with num_beams={num_beams} and lr={lr}")

        max_tok_len_trained = get_training_max_tok_len(task_prefix, cfg_item["tokenizer"])
        inputs, preds, labels = evaluate_model(model, testset_opt_trained_batched, cfg_item["tokenizer"], generation_config_opt, max_tok_len_trained, num_beams=num_beams)

        results = compute_metrics((inputs, preds, labels), cfg_item["tokenizer"])
        results["model_name"] = name
        results["num_beams"] = num_beams

        results_list.append(results)
        
        print(results)
        
    del model
    del inputs
    del preds
    del labels
    
    torch.cuda.empty_cache()
        
    gc.collect()

df_results = pd.DataFrame(results_list)
df_results.to_csv("results_opt_finetuned.csv", index=False)
print("Results saved to results_opt_finetuned.csv")


