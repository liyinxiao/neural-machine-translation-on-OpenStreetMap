# neural-machine-translation-on-OpenStreetMap

This work aims to apply transfer learning, and fine tune a pretrained hugging-face translation model to better translate the names of any map objects on OpenStreetMap (OSM), from Chinese (zh) to English (en). 
- Hugging Face pretrained model: https://huggingface.co/Helsinki-NLP/opus-mt-zh-en
- The OSM objects in the dataset are located in Taiwan (where native language is Chinese).

Setup
```
python3 -m venv .env
source .env/bin/activate

pip3 install transformers
pip3 install datasets
pip3 install evaluate
pip3 install sentencepiece
pip3 install torch torchvision
pip3 install sacrebleu
pip3 install sacremoses
```

Fine-tune a pre-trained Hugging Face model: https://huggingface.co/Helsinki-NLP/opus-mt-zh-en
```
python3 model_fine_tuning.py
```

Translate with the fine tuned model
```
python3 model_predict.py
```


### Results

The table below shows the translation results for 50 randomly selected objects. In general, the machine generated translations showed similar qualities to human-generated grouth truth data. The model fine tuned for 5 epochs performed slightly better than the one fine tuned for 1 epoch. 

| Raw text | Ground truth translation | Fine tuned model translation (1 epoch) | Fine tuned model translation (5 epochs) | Hugging Face model translation |
| ---------| ------------------------ | -------------------------------------- | --------------------------------------- | ------------------------------ |
| 北坑口 | Beikengkou | Beikengkou | Beikengkou | The North Puddle.
| 豐原大道八段 | Fengyuan Boulevard Section 8 | Fengyuan Boulevard Section 8 | Fengyuan Boulevard Section 8 | 8th section of Toyohara Avenue
| 長興里 | Changxing Village | Changxing Village | Changxing Village | ♪ In the long life ♪
| 建泰里 | Jiantai Village | Jiantai Village | Jiantai Village | To build Terry.
| 水源入口 | Shuiyuan Entrance | Shuiyuan Entrance | Shuiyuan Entrance | Water entrance.
| 仁義街178巷 | Lane 178, Renyi Street | Lane 178, Renyi Street | Lane 178, Renyi Street | 178th Street, Kinderty Street.
| 中華語文研習所 | Taipei Language Institute | Zhonghua Language Research Institute | Chinese Language Research Institute | Chinese Language Research Institute
| 花蓮機場 | Hualien Airport | Hualien Airport | Hualien Airport | Florist Airport.
| 博愛街 | Bo ai Street | Bo'ai Street | Bo'ai Street | Breath Street.
| 高雄市私立立志高級中學 | Li-Chih Valuable School | Kaohsiung Municipal Chihsiung Senior High School | Kaohsiung Municipal Li Zhigao Senior High School | It's a private high | school in Koshio City.
| Starbucks | Starbucks | Starbucks | Starbucks | Starbucks
| 海平路 | Hai-ping Road | Haiping Road | Haiping Road | It's a sea level road.
| 德民路356巷60弄 | Aly. 60, Lane 356, Demin Road | Alley 60, Lane 356, Demin Road | Alley 60, Lane 356, Demin Road | It's 60 on 356 Avenue, Demin Road.
| 南昌街 | Nanchang Street | Nanchang Street | Nanchang Street | Nanchang Street.
| 大甲國中 | Dajia Junior High School | Dajia Junior High School | Dajia Junior High School | I'm in the middle of the country.
| 長安街口 | Changan St. Intersection | Chang'an St. Intersection | Chang’an Street Intersection | Chang An Street.
| 元大商業銀行 | Yuanta Commercial Bank | Yuanda Commercial Bank | Yuan Da Commercial Bank | Yuan Chamber of Commerce and Industry
| 烏日戶政 | Wuri Household Reg. Office | Wuri Household Registration | Wuri Household Registration Office | Uzhi-ju.
| 台北 | Taipei | Taipei | Taipei | Taipei
| 協和東路 | Xiehe East Road | Xiehe East Road | Xiehe East Road | Union and East Road.
| 大豐山 | Dafengshan | Dafengshan | Dafengshan | The mountain of Toyo.
| 忠明五街 | Zhongming 5th Street | Zhongming 5th Street | Zhongming 5th Street | Jung-ming Fifth Street.
| 內門區公所 | Neimen District Office | Neimen District Office | Neimen District Office | The Inner District.
| 銅安山 | Tonganshan | Tong'anshan | Tong'anshan | Copper Ansan.
| 山脚 | Shanjiao | Shanjiao | Shanjiao | The foot of the mountain.
| 和平東路二段76巷23弄 | Alley 23, Lane 76, Section 2, Heping East Road | Alley 23, Lane 76, Section 2, Heping East Road | Alley 23, Lane 76, Section 2, Heping East Road | | Two-six-six-thirty-three on East and Peace Road.
| 先鋒路 | Xianfeng Road | Qianfeng Road | Xianfeng Road | Let's go first.
| 信義里 | Xinyi Village | Xinyi Village | Xinyi Village | Faithful.
| 青埔村 | Qingpu Village | Qingpu Village | Qingpu Village | Aoki Village
| 建國南路二段 | Section 2, Jianguo South Road | Section 2, Jianguo South Road | Section 2, Jianguo South Road | We're building two sections of South Road.
| 捷運大安森林公園站 | MRT Daan Park Sta. | MRT Da'an Forest Park Station | MRT Daan Forest Park Station | Da An Forest Park Station.
| 建德路 | Jiande Road | Jiande Road | Jiande Road | Jedree Road.
| 樹梅坑 | Shumeikeng | Shumeikeng | Shumeikeng | A puddle of prunes.
| 平鎮和平公園 | Pingzhen Heping Park | Pingjhen Heping Park | Pingzhen Heping Park | Peace Park, Pilgrim.
| 豐興路二段 | Fengsing Road Section 2 | Section 2, Fengxing Road | Fengxing Road Section 2 | Two parts of the road.
| 忠誠路口 | Zhongcheng Rd. Entrance | Zhongcheng Road Intersection | Zhongcheng Road Intersection | Faithful Pass.
| 東山路 | Dongshan Road | Dongshan Road | Dongshan Road | East Mountain Road.
| 見晴國小 | Jian Ching Elementary School | Zhenqing Elementary School | Jianqing Elementary School | I'll see you soon.
| 溪底(三美路) | Xidi(SanMei Road) | Xidi(Sanmei Road) | Xidi(Sanmei Road) | The bottom of the stream.
| 南京西路口(塔城) | Nanjing W. Rd. Entrance(Tacheng) | Nanjing W. Rd. Intersection (Tacheng) | Nanjing W. Rd. Intersection (Tacheng) | Nanjingxi Pass (Tata City)
| MOS Burger 摩斯漢堡 | MOS Burger | MOS Burger | MOS Burger | MOS Burger Mosburger
| 楠梓交流道 | Nanzi Interchange | Nanzi Interchange | Nanzi Interchange | Nanjing Concourse
| 外澳 | Wai'ao | Wai'ao | Wai'ao | Australia
| 芒果咖啡 | Mango Coffee | Mango Coffee | Mango Coffee | Mango coffee.
| Mo-Mo Paridise | Mo-Mo Paridise | Mo-Mo Paridishe | Mo-Mo Paridise | Mo-Mo Paridise
| 北華街 | Beihua Street | Beihua Street | Beihua Street | North China Street.
| 鹽菜坑 | Yancaikeng | Yancaikeng | Yancaikeng | Salt pothole.
| 新殿巷 | Xincheng Lane | Xincheng Lane | Xindian Lane | New Temple Lane.
| 大科路 | Dake Road | Dake Road | Dake Road | The main road.
| 新興路 | Xinxing Road | Xinxing Road | Xinxing Road | It's new.

This table below shows a few examples where machine generated translations are better than human generated ones. 

| Raw text | Ground truth translation | Fine tuned model translation (5 epoch) |
| ---------| ------------------------ | -------------------------------------- |
| 校前街40巷 | Lane 39, Xiaoqian Street | Lane 40, Xiaoqian Street | 
| 九天宮 | Jioutian Temple | Jiutian Temple | 
| 成功二路 | ChengGong 2nd Road | Chenggong 2nd Road | 
| 育群街 | Yucun Street | Yuqun Street | 
| 創義麵 | Creative Pasta | Creative Noodles | 
| 豐勢路 | Fongshih Road | Fengshi Road | There's a lot going on.
| 牛埔子 | Niubuzi | Niupuzi | Cow pebbles.

This table below shows a few examples where machine generated translations are worse than human generated ones. 

| Raw text | Ground truth translation | Fine tuned model translation (5 epoch) |
| ---------| ------------------------ | -------------------------------------- |
| 壟鉤路 | Longgou Road | Wucuo Road | I'm on my way.
| 海豐堀 | Haifengku | Haifengcuo | The sea is full.
| 日出茶太 | Chatime | Sunrise Tea Tai | Sunrise Tea too
| 沙灘小酒館 | beach bistro | Sha-Tsai Wine | The beach tavern.
| 社寮岡 | Sheliaogang | Shecuo | I'll take care of it.
| 奧萬大森林遊樂區 | Aowa Forest Recreation Area | Aowanda Forest Recreation Area | The O'Wan Great Forest Rover.
| 九兄弟 | Jiuxiongdi | Jiujiujiu | Nine brothers.
| 購物廣場站 | Shopping Plaza Station | Mall Mall | The mall.
| 巨上國際 | JApple Design | Kaohsiung International | Big international.


### Conclusion

The fine tuned model after 1 epoch already worked much better than the original Hugging Face model, and its performance further improved after 5 epochs of training. Compared with human-generated translation data in OpenStreetMap, the fine tuned model performs similarly, or slightly worse. However, this can be further improved by fine tuning with more map-oriented data, or starting with a better pre-trained model. 