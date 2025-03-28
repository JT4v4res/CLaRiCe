import os
import string
import unidecode
import tensorflow as tf
from tensorflow.keras import models, layers, metrics, losses, optimizers
from tensorflow.keras import callbacks
import numpy as np
from utils.utilities import *
import nltk
import spacy
from transformers import AutoConfig, TFAutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')

def model_load():
    clarice = tf.saved_model.load('./pred/model/conv1d_7_study_3/conv1d_7_study_3/')

    return clarice.signatures["serving_default"]


def get_lemmas(text: str):
    nlp = spacy.load("pt_core_news_lg")

    lemmas = []

    for t in text:
        doc = nlp(t)

        lemmas.append(' '.join(token.lemma_ for token in doc))

    return lemmas


def bert_tokenizer(text: str):
    tokens = tokenizer(text, padding="max_length", truncation=True, max_length=128)['input_ids']

    return tf.constant(tokens)


def preprocess_text(text):
    try:
        nltk.download('stopwords')
    except:
        print('stopwords already downloaded')

    # Obtenha a lista de stopwords e pontuações (caso queira usá-las para filtrar)
    stopwords = nltk.corpus.stopwords.words("portuguese")
    puncts = list(string.punctuation)
    punct_stopwords = puncts + stopwords

    # Tokenize o texto original
    tokens = nltk.tokenize.WordPunctTokenizer().tokenize(text)

    # Converte para minúsculo, remove acentos e (opcionalmente) filtra stopwords/pontuações
    processed_tokens = [unidecode.unidecode(token.lower()) for token in tokens]

    # Obtenha os lemas para cada token
    lemmas = get_lemmas(processed_tokens)

    # Una todos os lemas em uma única string
    joined_text = " ".join(lemmas)

    # Tokenize o texto unificado, garantindo que o comprimento seja 128 (com truncamento/padding)
    input_ids = bert_tokenizer(joined_text)

    return input_ids


def tf_predict(text):
    text = preprocess_text(text)  # Agora retorna um array com forma (128,)
    model = model_load()  # Carrega o modelo

    text = tf.expand_dims(text, 0)

    result = model(input_token=text)
    print(result)

    result_serializable = [float(tensor.numpy()) for tensor in result.values()]

    return {"predict_grades": result_serializable}

