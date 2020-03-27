import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer


def tokenize_corpus(corpus, tokenizer):
    tokenized_corpus = tokenizer.texts_to_sequences(corpus)
    return tokenized_corpus


def create_vocabulary(corpus, num_words=50000):
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(corpus)
    word2idx = tokenizer.word_index
    idx2word = {}
    for w in word2idx:
        idx2word[word2idx[w]] = w

    return list(word2idx.keys()), word2idx, idx2word, tokenizer
    # vocabulary = {}
    # for row in corpus:
    #     tokens = row.lower().strip().split()
    #     for tok in tokens:
    #         try:
    #             vocabulary[tok] += 1
    #         except:
    #             vocabulary[tok] = 1
    #
    # delete_keys = []
    # for i in vocabulary:
    #     if vocabulary[i] < min_count:
    #         delete_keys.append(i)
    #
    # for key in delete_keys:
    #     del vocabulary[key]
    #
    # word2idx = {}
    # idx2word = {}
    # count = 0
    # for i in vocabulary:
    #     word2idx[i] = count
    #     idx2word[count] = i
    #     count += 1

    # return vocabulary, word2idx, idx2word


def update_vocab(label_auth_dict, vocabulary, word2idx, idx2word):
    vocab_size = max(list(idx2word.keys())) + 1

    auth_set = set()
    for l in label_auth_dict:
        auth_set.update(set(label_auth_dict[l]))

    count = vocab_size
    for aut in auth_set:
        word2idx[aut] = count
        idx2word[count] = aut
        vocabulary.append(aut)
        count += 1
    return vocabulary, word2idx, idx2word


def create_label_auth_dict(auth_data_path, labels, label_topk_dict):
    label_auth_dict = {}
    for l in labels:
        top_auths_list = pickle.load(open(auth_data_path + l + "_top_auths.pkl", "rb"))
        label_auth_dict[l] = top_auths_list[:label_topk_dict[l]]
    return label_auth_dict
