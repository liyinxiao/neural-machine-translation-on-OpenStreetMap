from transformers import MarianMTModel, MarianTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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


# Hugging Face model
huggingface_model_name = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(huggingface_model_name)
translated = model.generate(
    **tokenizer(text, return_tensors="pt", padding=True))
huggingface_translated_text = [tokenizer.decode(
    t, skip_special_tokens=True) for t in translated]

# Hugging Face model with fine tuning
fine_tuned_model_name = 'opus-mt-zh-en-finetuned-zh-to-en'
tokenizer = MarianTokenizer.from_pretrained(fine_tuned_model_name)
model = MarianMTModel.from_pretrained(fine_tuned_model_name)
translated = model.generate(
    **tokenizer(text, return_tensors="pt", padding=True))
fine_tuned_translated_text = [tokenizer.decode(
    t, skip_special_tokens=True) for t in translated]

print("*************** Results *******************")
print("Raw text | Ground truth translation | Fine tuned model translation | Hugging Face model translation")
for zh, en, fine_tuned_translated_en, huggingface_translated_en in zip(text, ground_truth, fine_tuned_translated_text, huggingface_translated_text):
    print(zh + " | " + en + " | " + fine_tuned_translated_en +
          " | " + huggingface_translated_en)
