from transformers import MarianMTModel, MarianTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset

# Load test dataset
raw_datasets = load_dataset(
    "json", data_files={"test": "test_data.json"}, field="data")
print('***** Print raw_datasets *****')
print(raw_datasets["test"])

text = []
ground_truth_translated_text = []
for data in raw_datasets['test']:
    text.append(data["translation"]['zh'])
    ground_truth_translated_text.append(data["translation"]['en'])

# Create a subset of data for faster inference
text = text[0:50]
ground_truth_translated_text = ground_truth_translated_text[0:50]

# Hugging Face model prediciton
huggingface_model_name = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(huggingface_model_name)
translated = model.generate(
    **tokenizer(text, return_tensors="pt", padding=True))
huggingface_translated_text = [tokenizer.decode(
    t, skip_special_tokens=True) for t in translated]

# Fine tuned Hugging Face model prediction
fine_tuned_model_name = 'opus-mt-zh-en-finetuned-osm'
tokenizer = MarianTokenizer.from_pretrained(fine_tuned_model_name)
model = MarianMTModel.from_pretrained(fine_tuned_model_name)
translated = model.generate(
    **tokenizer(text, return_tensors="pt", padding=True))
fine_tuned_translated_text = [tokenizer.decode(
    t, skip_special_tokens=True) for t in translated]

print("*************** Results *******************")
print("Raw text | Ground truth translation | Fine tuned model translation | Hugging Face model translation |")
print("---------| ------------------------ | ---------------------------- | ------------------------------ |")
for zh, grounnd_truth_en, fine_tuned_en, huggingface_en in zip(text, ground_truth_translated_text, fine_tuned_translated_text, huggingface_translated_text):
    if grounnd_truth_en != fine_tuned_en:
        print(zh + " | " + grounnd_truth_en + " | " + fine_tuned_en +
              " | " + huggingface_en)


# Fine tuned Hugging Face model prediction (1 epoch)
fine_tuned_model_name = 'opus-mt-zh-en-finetuned-osm-1-epoch'
tokenizer = MarianTokenizer.from_pretrained(fine_tuned_model_name)
model = MarianMTModel.from_pretrained(fine_tuned_model_name)
translated = model.generate(
    **tokenizer(text, return_tensors="pt", padding=True))
fine_tuned_translated_text_1_epoch = [tokenizer.decode(
    t, skip_special_tokens=True) for t in translated]


print("*************** Results *******************")
print("Raw text | Ground truth translation | Fine tuned model translation (1 epoch) | Fine tuned model translation (5 epochs) | Hugging Face model translation |")
print("---------| ------------------------ | -------------------------------------- | --------------------------------------- | ------------------------------ |")
for zh, grounnd_truth_en, fine_tuned_en_1_epoch, fine_tuned_en, huggingface_en in zip(text, ground_truth_translated_text, fine_tuned_translated_text_1_epoch, fine_tuned_translated_text, huggingface_translated_text):
    print(zh + " | " + grounnd_truth_en + " | " + fine_tuned_en_1_epoch + " | " + fine_tuned_en +
          " | " + huggingface_en)
