import re
import numpy as np
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import pickle
from nltk import word_tokenize


def clean(data):
    # Remove new line characters
    data = [re.sub('\s+', ' ', sent) for sent in data]
    # Remove distracting single quotes
    data = [re.sub("\'", "", sent) for sent in data]
    return data


def custom_tokenize(text):
    PAT_ALPHABETIC = re.compile(r'((\w)+)', re.UNICODE)
    for match in PAT_ALPHABETIC.finditer(text):
        yield match.group()


def preprocess(doc):
    tokens = [
        token for token in custom_tokenize(doc)
    ]
    return tokens


def sent_to_words(sentences):
    for sentence in sentences:
        yield (preprocess(str(sentence)))


def remove_stopwords(texts):
    return [[word for word in preprocess(str(doc)) if word not in stop_words] for doc in texts]


if __name__ == "__main__":
    data_path = "./data/"

    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 'fnust218'])

    df = pickle.load(open(data_path + "df_29k_abstract_phrase.pkl", "rb"))
    data = df.abstract.values.tolist()
    data = clean(data)
    data_words = list(sent_to_words(data))
    data_words_nostops = remove_stopwords(data_words)

    id2word = corpora.Dictionary(data_words_nostops)
    texts = data_words_nostops
    corpus = [id2word.doc2bow(text) for text in texts]

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=10,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
    pprint(lda_model.print_topics())

    print('\nPerplexity: ', lda_model.log_perplexity(corpus))
