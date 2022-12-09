import os
import re
import nltk
import sys

from nltk.corpus import stopwords
from transformers import BertTokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from model.bert import BERT
from flask import Flask, render_template, request

nltk.download('stopwords')

app = Flask(__name__)

class Inference():
    def __init__(self, 
                model='checkpoints/indobert_result/epoch=2-step=744.ckpt'): # change the path if found better model

        if not os.path.exists(model):
            print("No model available, run trainer.py first to create model")
            sys.exit()

        self.model = BERT.load_from_checkpoint(model)
        self.model.eval()
        self.model.freeze()
        self.labels = ['negatif', 'positif']
        self.stop_words = stopwords.words('indonesian')
        self.stemmer = StemmerFactory().create_stemmer()
        self.tokenizer = BertTokenizer.from_pretrained('indolem/indobert-base-uncased')

    def predict(self, text):
        text = text.lower()
        text = re.sub(r"[^A-Za-z0-9(),!?\'\-`]", " ", text)
        text = re.sub(r"\'s", " \'s", text)
        text = re.sub(r"\'ve", " \'ve", text)
        text = re.sub(r"n\'t", " n\'t", text)
        text = re.sub(r"\n", "", text)
        text = re.sub(r"\'re", " \'re", text)
        text = re.sub(r"\'d", " \'d", text)
        text = re.sub(r"\'ll", " \'ll", text)
        text = re.sub(r",", " , ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\(", " \( ", text)
        text = re.sub(r"\)", " \) ", text)
        text = re.sub(r"\?", " \? ", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = text.strip()
        text = ' '.join([word for word in text.split() if word not in self.stop_words])
        text = self.stemmer.stem(text)   

        encoder = self.tokenizer.encode_plus(text=text,
                                            add_special_tokens=True,
                                            max_length=100,
                                            return_token_type_ids=True,
                                            padding="max_length",
                                            truncation=True,
                                            return_attention_mask=True,
                                            return_tensors='pt')

        output = self.model(encoder["input_ids"],
                        encoder["token_type_ids"],
                        encoder["attention_mask"])

        pred = output.argmax(1)
        # pred = output.argmax(1).cpu()

        return self.labels[pred]

classifier = Inference()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    text = request.form['text']
    result = classifier.predict(text)
    return render_template('index.html', prediction=result, text=text)

if __name__ == '__main__':
    app.run(debug=True, port=3000)
