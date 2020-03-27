import numpy as np
from collections import defaultdict
import random
import matplotlib.pyplot as plt
from parse_autophrase_output import generate_name, decrypt


def get_label_term_json(path):
    import json
    dic = json.load(open(path, "r"))
    return dic


def modify_phrases(label_term_dict, phrase_id_map, random_k=0):
    for l in label_term_dict:
        temp_list = []
        for term in label_term_dict[l]:
            try:
                temp_list.append(generate_name(phrase_id_map[term]))
            except:
                temp_list.append(term)
        if random_k:
            random.shuffle(temp_list)
            label_term_dict[l] = temp_list[:random_k]
        else:
            label_term_dict[l] = temp_list
    return label_term_dict


def create_index(tokenizer):
    index_to_word = {}
    word_to_index = tokenizer.word_index
    for word in word_to_index:
        index_to_word[word_to_index[word]] = word
    return word_to_index, index_to_word


def print_label_term_dict(label_term_dict, components, id_phrase_map):
    for label in label_term_dict:
        print(label)
        print("*" * 80)
        for val in label_term_dict[label]:
            try:
                id = decrypt(val)
                if id is not None and id in id_phrase_map:
                    phrase = id_phrase_map[id]
                    print(phrase, components[label][val])
                else:
                    print(val, components[label][val])
            except Exception as e:
                print("Exception occured: ", e, val)


def print_label_term_dict_direct(label_term_dict, components):
    for label in label_term_dict:
        print(label)
        print("*" * 80)
        for val in label_term_dict[label]:
            try:
                print(val, components[label][val])
            except Exception as e:
                print("Exception occured: ", e, val)


def print_label_phrase_dict(label_phrase_dict, id_phrase_map):
    for label in label_phrase_dict:
        print(label)
        print("*" * 80)
        for key in label_phrase_dict[label]:
            id = decrypt(key)
            print(id_phrase_map[id], label_phrase_dict[label][key])


def print_label_entity_dict(label_entity_dict):
    for label in label_entity_dict:
        print(label)
        print("*" * 80)
        for key in label_entity_dict[label]:
            print(key, label_entity_dict[label][key])


def get_term_freq(df):
    term_freq = defaultdict(int)
    for index, row in df.iterrows():
        line = row["sentence"]
        words = line.strip().split()
        for w in words:
            term_freq[w] += 1
    return term_freq


def calculate_doc_freq(docs):
    docfreq = {}
    for doc in docs:
        words = doc.strip().split()
        temp_set = set(words)
        for w in temp_set:
            try:
                docfreq[w] += 1
            except:
                docfreq[w] = 1
    return docfreq


def get_doc_freq(df):
    docfreq = {}
    docfreq["UNK"] = len(df)
    for index, row in df.iterrows():
        line = row["abstract"]
        words = line.strip().split()
        temp_set = set(words)
        for w in temp_set:
            try:
                docfreq[w] += 1
            except:
                docfreq[w] = 1
    return docfreq


def get_inv_doc_freq(df, docfreq):
    inv_docfreq = {}
    N = len(df)
    for word in docfreq:
        inv_docfreq[word] = np.log(N / docfreq[word])
    return inv_docfreq


def get_label_docs_dict(df, label_term_dict, pred_labels):
    label_docs_dict = {}
    for l in label_term_dict:
        label_docs_dict[l] = []
    for index, row in df.iterrows():
        line = row["abstract"]
        label_docs_dict[pred_labels[index]].append(line)
    return label_docs_dict


def plot_hist(values, bins):
    plt.figure()
    n = plt.hist(values, color='blue', edgecolor='black', bins=bins)
    print(n)
    plt.show()


def modify_df(df, docfreq, threshold=5):
    UNK = "UNK"
    for index, row in df.iterrows():
        line = row["sentence"]
        words = line.strip().split()
        for i, w in enumerate(words):
            if docfreq[w] < threshold:
                words[i] = UNK
        df["sentence"][index] = " ".join(words)
    return df
