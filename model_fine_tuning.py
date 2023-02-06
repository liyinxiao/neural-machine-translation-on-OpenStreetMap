# Reference:
# https://medium.com/@tskumar1320/how-to-fine-tune-pre-trained-language-translation-model-3e8a6aace9f
# https://github.com/sravan1320/NMT/blob/main/fine_tune_hugging_face_translation_model.ipynb
# https://neptune.ai/blog/hugging-face-pre-trained-models-find-the-best

from datasets import load_dataset
import evaluate
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, MarianMTModel, MarianTokenizer, AutoTokenizer
import numpy as np

# Import the pre-trained Hugging Face model
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")

# Load the dataset extracted from OpenStreetMap
raw_datasets = load_dataset("json", data_files={
                            "train": "train_data.json", "test": "test_data.json"}, field="data")
print('***** Print raw_datasets *****')
print(raw_datasets)

# Preprocess the dataset
prefix = ""
max_input_length = 64
max_target_length = 64
source_lang = "zh"
target_lang = "en"


def preprocess_function(examples):
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(
        inputs, max_length=max_input_length, truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
print('***** Print tokenized_datasets *****')
print(tokenized_datasets)

# Create a subset of data for faster training
# tokenized_datasets["train"] = tokenized_datasets["train"].shuffle(
#     seed=42).select(range(1000))
# tokenized_datasets["test"] = tokenized_datasets["test"].shuffle(
#     seed=42).select(range(20))

# Fine tune the model
batch_size = 16
model_name = "opus-mt-zh-en-finetuned-osm"
args = Seq2SeqTrainingArguments(
    model_name,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True
)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
metric = evaluate.load("sacrebleu")


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
    decoded_preds, decoded_labels = postprocess_text(
        decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds,
                            references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(
        pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


# Call trainer API and start fine tuning
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

# Save the model
trainer.save_model(model_name)

# Evaluate on test data
text = []
ground_truth_translated_text = []
for data in raw_datasets['test']:
    text.append(data["translation"]['zh'])
    ground_truth_translated_text.append(data["translation"]['en'])
text = text[0:50]
ground_truth_translated_text = ground_truth_translated_text[0:50]

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
translated = model.generate(
    **tokenizer(text, return_tensors="pt", padding=True))
translated_text = [tokenizer.decode(
    t, skip_special_tokens=True) for t in translated]

print("*************** Results *******************")
print("Raw text | Ground truth translation | Fine tuned model translation")
for zh, en, translated_en in zip(text, ground_truth_translated_text, translated_text):
    print(zh + " | " + en + " | " + translated_en)
