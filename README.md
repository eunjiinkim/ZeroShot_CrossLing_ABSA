# Zero-shot Cross-lingual Aspect-Based Sentiment Analysis through a Template Re-generation Task


### The codes should be fixed!

Using [mBART-large-50](https://huggingface.co/facebook/mbart-large-50), we make the model fill the discrete templates with aspect and sentiment for zero-shot cross-lingual learning. \
The source language is English and the target languages are Spanish, French, Russian, and Dutch. \
The below figure is an example in Spanish. \
\
<img width="60%" alt="스크린샷 2022-12-29 오후 10 30 11" src="https://user-images.githubusercontent.com/55074165/209960395-112984d8-b42e-44be-bb14-2a0d04f71943.png">
\
* The input sentence is in English, but we use the code-switched templates so that two inputs in both languages are fed to the model.
* Then, we compute the cross-entropy losses and updates the parameters by averaging them.

## Dataset
* We use SemEval2016 restaurant ABSA dataset.
* Since the codes in [process_xlm_json_tsv-bioes.ipynb] contain some errors, we recommend to use [process_data.ipynb]
* Here, we make the labels in BIOES tagging. 

## Methods
\
<img width="827" alt="스크린샷 2022-12-29 오후 10 33 39" src="https://user-images.githubusercontent.com/55074165/209961136-6c5144bc-0369-4862-83f9-a0df09167b1d.png">
 
 * Examples for each template in English. 
 * Input sentence is ‘Space was limited, but the food made up for it’ and target pairs are (space, negative) and (food, positive). \
 ```
 python3 train_{}_bilingual.py --lang {target_lang} 
 ```
