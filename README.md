# neural-machine-translation-on-OpenStreetMap

This work aims to fine tune a pretrained hugging-face translation model to translate the names of ways on OpenStreetMap (OSM), from Chinese (zh) to English (en). 
- Hugging Face pretrained model: https://huggingface.co/Helsinki-NLP/opus-mt-zh-en
- The OSM ways in the dataset are located in Taiwan, have one of the following tags:
    - ```
        tags['highway'] IN (
            'motorway',
            'trunk',
            'primary',
            'secondary',
            'tertiary',
            'residential',
            'unclassified',
            'road',
            'living_street',
            'service',
            'footway',
            'path',
            'pedestrian',
            'bridleway',
            'cycleway',
            'track',
            'steps'
        )
        ```

Setup
```
python3 -m venv  .env
source .env/bin/activate

pip install transformers
pip3 install datasets
pip3 install evaluate
pip3 install sentencepiece
pip3 install torch torchvision
pip3 install sacrebleu
pip3 install sacremoses
```

Fine-tune a pre-trained Hugging Face model: https://huggingface.co/Helsinki-NLP/opus-mt-zh-en
```
python3 fine_tune_huggingface_translation_model.py
```

Translate with the fine tuned model
```
python3 model_predict.py
```


### Results (after 2 epochs of fine tuning)

| Raw text | Ground truth translation | Fine tuned model translation | Hugging Face model translation|
| -------- | ------------------------ | ---------------------------- | ----------------------------- |
| 農園路 | Nongyuan Road | Nongyuan Road | Campus Road.
| 拕子南街 | Tuozi South Street | Shanzi South Street | South Street.
| 羅斯福路六段142巷 | Lane 142, Roosevelt Road Section 6 | Lane 142, Section 6, Roosevelt Road | Six hundred and forty-two on Roosevelt Road.
| 明湖路 | Minghu Road | Minghu Road | Ming Lake Road
| 海專路 | Hai-jhuan Road | Haichou Road | Seaworks.
| 台北聯絡線 | Taipei Connection Line | Taipei Line of Contact | Taipei contact.
| 建國北路一段 | Section 1, Jianguo North Road | Section 1, Jianguo North Road | Building North Road for a while.
| 協中街211巷 | Lane 211,Xiezhong Street | Lane 211, Xizhong Street | 211th Street, Union Center.
| 大同路77巷 | Lane 77, Datong Road | Lane 77, Datong Road | 77th Avenue.
| 福山二路 | Fushan 2nd Road | Fushan 2nd Road | Fukuyama 2nd Road.
| 羅斯福路五段218巷32弄 | Alley 32, Lane 218, Section 5, Roosevelt Road | Alley 32, Lane 218, Section 5, Roosevelt Road | Five blocks of 218 on Roosevelt Road. 32 on Rosford | Road.
| 國光路 | Guoguang Road | Guoguang Road | The country's light.
| 景新街467巷32弄 | Alley 32, Lane 467, Jingxin Street | Alley 32, Lane 467, Jingxin Street | Four hundred and sixty-two on Cygen Street.
| 永明街 | Yongming Street | Yongming Street | Young-ming Street.
| 建功巷 | Jiangong Lane | Jiangong Lane | Build an alley.
| 中華路二段39巷 | Lane 39, Section 2, Zhonghua Road | Lane 39, Section 2, Zhonghua Road | Two blocks of 39 on China Road.
| 延平北路四段 | Yanping North Road Section 4 | Yanping North Road Section 4 | Four blocks north on Yanping Road.


### Conclusion

The fine tuned model works much better than the original Hugging Face model, but it still gets a few translations wrong. 