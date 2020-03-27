from sklearn.feature_extraction.text import CountVectorizer
from coc_data_utils import *
import numpy as np
import math


def get_popular_matrix(index_to_word, docfreq, inv_docfreq, label_count, label_docs_dict, label_to_index,
                       term_count, word_to_index, doc_freq_thresh):
    E_LT = np.zeros((label_count, term_count))
    components = {}
    for l in label_docs_dict:
        components[l] = {}
        docs = label_docs_dict[l]
        docfreq_local = calculate_doc_freq(docs)
        vect = CountVectorizer(vocabulary=list(word_to_index.keys()), tokenizer=lambda x: x.split())
        X = vect.fit_transform(docs)
        X_arr = X.toarray()
        rel_freq = np.sum(X_arr, axis=0) / len(docs)
        names = vect.get_feature_names()
        for i, name in enumerate(names):
            try:
                if docfreq_local[name] < doc_freq_thresh:
                    continue
            except:
                continue
            E_LT[label_to_index[l]][word_to_index[name]] = (docfreq_local[name] / docfreq[name]) * inv_docfreq[name] \
                                                           * np.tanh(rel_freq[i])
            components[l][name] = {"reldocfreq": docfreq_local[name] / docfreq[name],
                                   "idf": inv_docfreq[name],
                                   "rel_freq": np.tanh(rel_freq[i]),
                                   "rank": E_LT[label_to_index[l]][word_to_index[name]]}
    return E_LT, components


def update(E_LT, F_LT, index_to_label, index_to_word, it, label_count, n1, n2, label_docs_dict):
    word_map = {}
    for l in range(label_count):
        if not np.any(E_LT):
            n = 0
        else:
            n = min(n1 * (it + 1), int(math.log(len(label_docs_dict[index_to_label[l]]), 1.5)))
        inds_popular = E_LT[l].argsort()[::-1][:n]

        if not np.any(F_LT):
            n = 0
        else:
            n = min(n2 * (it + 1), int(math.log(len(label_docs_dict[index_to_label[l]]), 1.5)))
        inds_exclusive = F_LT[l].argsort()[::-1][:n]

        for word_ind in inds_popular:
            word = index_to_word[word_ind]
            try:
                temp = word_map[word]
                if E_LT[l][word_ind] > temp[1]:
                    word_map[word] = (index_to_label[l], E_LT[l][word_ind])
            except:
                word_map[word] = (index_to_label[l], E_LT[l][word_ind])

        for word_ind in inds_exclusive:
            word = index_to_word[word_ind]
            try:
                temp = word_map[word]
                if F_LT[l][word_ind] > temp[1]:
                    word_map[word] = (index_to_label[l], F_LT[l][word_ind])
            except:
                word_map[word] = (index_to_label[l], F_LT[l][word_ind])

    label_term_dict = defaultdict(set)
    for word in word_map:
        label, val = word_map[word]
        label_term_dict[label].add(word)
    return label_term_dict


def update_label_term_dict(df, label_term_dict, pred_labels, label_to_index, index_to_label, word_to_index,
                           index_to_word, inv_docfreq, docfreq, it, n1, n2, doc_freq_thresh=5):
    label_count = len(label_to_index)
    term_count = len(word_to_index)
    label_docs_dict = get_label_docs_dict(df, label_term_dict, pred_labels)

    E_LT, components = get_popular_matrix(index_to_word, docfreq, inv_docfreq, label_count, label_docs_dict,
                                          label_to_index, term_count, word_to_index, doc_freq_thresh)
    F_LT = np.zeros((label_count, term_count))

    label_term_dict = update(E_LT, F_LT, index_to_label, index_to_word, it, label_count, n1, n2, label_docs_dict)
    return label_term_dict, components


def update_label_entity_dict_with_iteration(label_entity_dict, df, pred_labels, it, n1=7):
    label_docs_dict = get_label_docs_dict(df, label_entity_dict, pred_labels)
    for l in label_entity_dict:
        n = min(n1 * (it + 1), int(math.log(len(label_docs_dict[l]), 1.5)))
        i = 0
        top_k = {}
        for tup in list(label_entity_dict[l].items()):
            if i >= n:
                break
            top_k[tup[0]] = tup[1]
            i += 1
        label_entity_dict[l] = top_k
    return label_entity_dict


def update_label_conf_dict(label_conf_dict, it):
    n = min(it + 1, 3)
    for l in label_conf_dict:
        top_k = {}
        i = 0
        for tup in list(label_conf_dict[l].items()):
            if i >= n:
                break
            top_k[tup[0]] = tup[1]
            i += 1
        label_conf_dict[l] = top_k
    return label_conf_dict


def update_by_percent_with_overlap(label_entity_dict, entity_docid_map, df, i):
    filtered_dict = {}
    for l in label_entity_dict:
        filtered_dict[l] = {}

    # n = min((i + 1) * 0.1 * len(df), len(df) * 0.6)
    n = min((i + 1) * 0.1 * len(df), len(df))
    doc_id_set = set()
    i = 0
    while len(doc_id_set) < n:
        for l in label_entity_dict:
            all_tups = list(label_entity_dict[l].items())
            if i < len(all_tups):
                tup = all_tups[i]
                filtered_dict[l][tup[0]] = tup[1]
                doc_id_set.update(entity_docid_map[tup[0]])
        i += 1
    return filtered_dict


def update_by_percent(label_phrase_dict, phrase_docid_map, df, i):
    filtered_dict = {}
    for l in label_phrase_dict:
        filtered_dict[l] = {}

    # n = min((i + 1) * 0.1 * len(df), len(df) * 0.6)
    n = min((i + 1) * 0.1 * len(df), len(df))
    checked_phrases = {}
    doc_id_set = set()
    i = 0
    while len(doc_id_set) < n:
        for l in label_phrase_dict:
            all_tups = list(label_phrase_dict[l].items())
            if i < len(all_tups):
                tup = all_tups[i]
                try:
                    temp_ph = checked_phrases[tup[0]]
                    return filtered_dict
                except:
                    filtered_dict[l][tup[0]] = tup[1]
                    checked_phrases[tup[0]] = 1
                    doc_id_set.update(phrase_docid_map[tup[0]])
        i += 1
    return filtered_dict


def update_by_percent_together(label_entity_dict_list, entity_docid_map_list, df, labels, i, cov="full"):
    filtered_label_entity_dict_list = []
    for label_entity_dict in label_entity_dict_list:
        filtered_dict = {}
        for l in label_entity_dict:
            filtered_dict[l] = {}
        filtered_label_entity_dict_list.append(filtered_dict)

    if cov == "full":
        n = len(df)
    else:
        n = min((i + 1) * 0.1 * len(df), len(df))
    doc_id_set = set()

    sorted_tups_dict = {}
    for l in labels:
        all_tups = []
        for label_entity_dict in label_entity_dict_list:
            all_tups += list(label_entity_dict[l].items())
        sorted_tups_dict[l] = sorted(all_tups, key=lambda tup: -tup[1])

    index = 0
    while len(doc_id_set) < n:
        flag = 0
        for l in labels:
            if index < len(sorted_tups_dict[l]):
                tup = sorted_tups_dict[l][index]
                for i, entity_docid in enumerate(entity_docid_map_list):
                    try:
                        temp = entity_docid[tup[0]]
                        filtered_label_entity_dict_list[i][l][tup[0]] = tup[1]
                        doc_id_set.update(entity_docid[tup[0]])
                        flag = 1
                        break
                    except:
                        continue
        if flag == 0:
            break
        index += 1
    return filtered_label_entity_dict_list
