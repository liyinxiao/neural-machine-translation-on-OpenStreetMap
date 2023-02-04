# Reference: 
# https://medium.com/@tskumar1320/how-to-fine-tune-pre-trained-language-translation-model-3e8a6aace9f
# https://github.com/sravan1320/NMT/blob/main/fine_tune_hugging_face_translation_model.ipynb

from datasets import load_dataset, load_metric
import evaluate
import numpy as np
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import MarianMTModel, MarianTokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

# Load and preprocess data
prefix = ""
max_input_length = 64
max_target_length = 64
source_lang = "zh"
target_lang = "en"
def preprocess_function(examples):
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

raw_datasets = load_dataset("json", data_files={"train": "train_data.json", "test": "test_data.json"}, field="data")
metric = evaluate.load("sacrebleu")
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
print('***** Print raw_datasets *****')
print(raw_datasets)
print('***** Print tokenized_datasets *****')
print(tokenized_datasets)

# Fine tune the model
batch_size = 16
model_name = "opus-mt-zh-en"
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-{source_lang}-to-{target_lang}",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    predict_with_generate=True
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
trainer.train()
trainer.save_model("opus-mt-zh-en-finetuned-zh-to-en")

print("*************** Results *******************")
text = [
    '農園路',
    '拕子南街',
    '羅斯福路六段142巷',
    '明湖路',
    '海專路',
    '台北聯絡線',
    '建國北路一段',
    '協中街211巷',
    '大同路77巷',
    '福山二路',
    '羅斯福路五段218巷32弄',
    '國光路',
    '景新街467巷32弄',
    '永明街',
    '建功巷',
    '中華路二段39巷',
    '延平北路四段',
]

ground_truth = [
    "Nongyuan Road",
    "Tuozi South Street",
    "Lane 142, Roosevelt Road Section 6",
    "Minghu Road",
    "Hai-jhuan Road",
    "Taipei Connection Line",
    "Section 1, Jianguo North Road",
    "Lane 211,Xiezhong Street",
    "Lane 77, Datong Road",
    "Fushan 2nd Road",
    "Alley 32, Lane 218, Section 5, Roosevelt Road",
    "Guoguang Road",
    "Alley 32, Lane 467, Jingxin Street",
    "Yongming Street",
    "Jiangong Lane",
    "Lane 39, Section 2, Zhonghua Road",
    "Yanping North Road Section 4",
]

model_name = 'opus-mt-zh-en-finetuned-zh-to-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
translated_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

print("Raw text | Ground truth translation | Fine tuned model translation")
for zh, en, translated_en in zip(text, ground_truth, translated_text):
    print(zh + " | " + en + " | " + translated_en)