from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras_han.model import HAN
from scipy.special import softmax
from keras.losses import kullback_leibler_divergence
import matplotlib.pyplot as plt
from data_utils import *
import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def create_training_df(X, y, y_true):
    dic = {}
    dic["Data"] = X
    dic["Training label"] = y
    dic["True label"] = y_true
    df_X = DataFrame(dic)
    return df_X


def get_distinct_labels(df):
    label_to_index = {}
    index_to_label = {}
    labels = set(df["label"])

    for i, label in enumerate(labels):
        label_to_index[label] = i
        index_to_label[i] = label
    return labels, label_to_index, index_to_label


def get_distinct_labels_from_label_term_dict(label_term_dict):
    label_to_index = {}
    index_to_label = {}
    labels = set(label_term_dict.keys())

    for i, label in enumerate(labels):
        label_to_index[label] = i
        index_to_label[i] = label
    return labels, label_to_index, index_to_label


def get_entity_count(label_entity_dict, entity_count):
    for l in label_entity_dict:
        try:
            entity_count[l].append(len(label_entity_dict[l]))
        except:
            entity_count[l] = [len(label_entity_dict[l])]
    return entity_count


def get_cut_off(label_entity_dict, entity_cut_off):
    for l in label_entity_dict:
        items_list = list(label_entity_dict[l].items())
        try:
            entity_cut_off[l].append(items_list[-1][1])
        except:
            entity_cut_off[l] = [items_list[-1][1]]
    return entity_cut_off


def plot_entity_count(y_values, x_values, path, x_label, y_label):
    plt.figure()
    plt.plot(x_values, y_values)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.savefig(path)


def argmax_label(count_dict):
    maxi = 0
    max_label = None
    for l in count_dict:
        count = 0
        for t in count_dict[l]:
            count += count_dict[l][t]
        if count > maxi:
            maxi = count
            max_label = l

    return max_label


def softmax_label(count_dict, label_to_index):
    temp = [0] * len(label_to_index)
    for l in count_dict:
        count = 0
        for t in count_dict[l]:
            count += count_dict[l][t]
        temp[label_to_index[l]] = count
    return softmax(temp)


def get_train_data(df, labels, label_term_dict, label_author_dict, tokenizer, label_to_index, ignore_metadata=True,
                   soft=False):
    y = []
    X = []
    index_word = {}
    for w in tokenizer.word_index:
        index_word[tokenizer.word_index[w]] = w
    for index, row in df.iterrows():
        authors_set = set(row["authors"])
        line = row["abstract"]
        tokens = tokenizer.texts_to_sequences([line])[0]
        words = []
        for tok in tokens:
            words.append(index_word[tok])
        count_dict = {}
        flag = 0
        for l in labels:
            seed_words = set(label_term_dict[l].keys())
            int_labels = list(set(words).intersection(seed_words))

            if len(label_author_dict) > 0:
                seed_authors = set(label_author_dict[l].keys())
                int_authors = authors_set.intersection(seed_authors)
            else:
                int_authors = []
            if ignore_metadata and len(int_labels) == 0:
                continue
            for word in words:
                if word in int_labels:
                    flag = 1
                    try:
                        temp = count_dict[l]
                    except:
                        count_dict[l] = {}
                    try:
                        count_dict[l][word] += label_term_dict[l][word]
                    except:
                        count_dict[l][word] = label_term_dict[l][word]

            # if flag:
            for auth in int_authors:
                try:
                    temp = count_dict[l]
                except:
                    count_dict[l] = {}
                count_dict[l]["AUTH_" + str(auth)] = label_author_dict[l][auth]
                flag = 1

        if flag:
            if not soft:
                lbl = argmax_label(count_dict)
                if not lbl:
                    continue
            else:
                lbl = softmax_label(count_dict, label_to_index)
            y.append(lbl)
            X.append(line)
    return X, y


def get_count_dict_metadata(authors_set, conf, label_author_dict, labels):
    count_dict = {}
    flag = 0
    for l in labels:
        if len(label_author_dict) > 0:
            seed_authors = set(label_author_dict[l].keys())
            int_authors = authors_set.intersection(seed_authors)
        else:
            int_authors = []

        for auth in int_authors:
            flag = 1
            try:
                temp = count_dict[l]
            except:
                count_dict[l] = {}
            count_dict[l]["AUTH_" + str(auth)] = label_author_dict[l][auth]

    return count_dict, flag


def get_count_dict_phrase(label_term_dict, labels, words):
    count_dict = {}
    flag = 0
    for l in labels:
        seed_words = set(label_term_dict[l].keys())
        int_labels = list(set(words).intersection(seed_words))
        for word in words:
            if word in int_labels:
                flag = 1
                try:
                    temp = count_dict[l]
                except:
                    count_dict[l] = {}
                try:
                    count_dict[l][word] += label_term_dict[l][word]
                except:
                    count_dict[l][word] = label_term_dict[l][word]
    return count_dict, flag


def get_phrase_label(words, label_term_dict, labels, label_to_index, soft=False):
    count_dict, flag = get_count_dict_phrase(label_term_dict, labels, words)
    return get_label_from_count_dict(count_dict, flag, label_to_index, soft=soft)


def get_metadata_label(authors_set, label_author_dict, conf, labels, label_to_index, soft=False):
    count_dict, flag = get_count_dict_metadata(authors_set, conf, label_author_dict, labels)
    return get_label_from_count_dict(count_dict, flag, label_to_index, soft)


def get_label_from_count_dict(count_dict, flag, label_to_index, soft):
    lbl = None
    if flag:
        if not soft:
            lbl = argmax_label(count_dict)
        else:
            lbl = softmax_label(count_dict, label_to_index)
    return lbl


def merge(count_dict_phrase, count_dict_metadata, labels):
    count_dict = {}
    flag = 0
    for l in labels:
        count_dict[l] = {}
        try:
            temp = count_dict_phrase[l]
            count_dict[l].update(count_dict_phrase[l])
        except:
            pass
        try:
            temp = count_dict_metadata[l]
            count_dict[l].update(count_dict_metadata[l])
        except:
            pass
        if len(count_dict[l]) > 0:
            flag = 1

    return count_dict, flag


def calculate_weight(l_phrase, l_metadata, label_index, AND=True):
    try:
        prob_phrase = l_phrase[label_index]
    except:
        prob_phrase = 0
    try:
        prob_metadata = l_metadata[label_index]
    except:
        prob_metadata = 0
    if AND:
        return prob_phrase * prob_metadata
    else:
        return prob_phrase + prob_metadata - prob_phrase * prob_metadata


def get_confident_train_data(df, labels, label_term_dict, label_author_dict, label_to_index, tokenizer):
    y = []
    y_phrase = []
    y_metadata = []
    X = []
    index_word = {}
    for w in tokenizer.word_index:
        index_word[tokenizer.word_index[w]] = w
    for index, row in df.iterrows():
        auth_str = row["authors"]
        authors_set = set(auth_str.split(","))
        conf = row["conf"]
        line = row["abstract"]
        label = row["label"]
        tokens = tokenizer.texts_to_sequences([line])[0]
        words = []
        for tok in tokens:
            words.append(index_word[tok])

        l_phrase = get_phrase_label(words, label_term_dict, labels, label_to_index)
        l_metadata = get_metadata_label(authors_set, label_author_dict, conf, labels, label_to_index)

        if l_phrase == l_metadata:
            y.append(l_phrase)
            X.append(line)
        elif l_phrase is None:
            y.append(l_metadata)
            X.append(line)
        elif l_metadata is None:
            y.append(l_phrase)
            X.append(line)

    return X, y


def get_weighted_train_data(df, labels, label_term_dict, label_author_dict, tokenizer, label_to_index, AND=True):
    y = []
    X = []
    y_true = []
    weights = []
    index_word = {}
    for w in tokenizer.word_index:
        index_word[tokenizer.word_index[w]] = w
    for index, row in df.iterrows():
        auth_str = row["authors"]
        authors_set = set(auth_str.split(","))
        conf = row["conf"]
        line = row["abstract"]
        label = row["label"]
        tokens = tokenizer.texts_to_sequences([line])[0]
        words = []
        for tok in tokens:
            words.append(index_word[tok])

        count_dict_phrase, flag_phrase = get_count_dict_phrase(label_term_dict, labels, words)
        l_phrase = get_label_from_count_dict(count_dict_phrase, flag_phrase, label_to_index, soft=True)

        count_dict_metadata, flag_metadata = get_count_dict_metadata(authors_set, conf, label_author_dict, labels)
        l_metadata = get_label_from_count_dict(count_dict_metadata, flag_metadata, label_to_index, soft=True)

        count_dict, flag = merge(count_dict_phrase, count_dict_metadata, labels)
        l_all = get_label_from_count_dict(count_dict, flag, label_to_index, soft=False)

        if l_all is None:
            continue
        weight = calculate_weight(l_phrase, l_metadata, label_to_index[l_all], AND=AND)
        y.append(l_all)
        X.append(line)
        y_true.append(label)
        weights.append(weight)
    return X, y, y_true, weights


def train_weight_classifier(df, labels, label_term_dict, label_author_dict, label_to_index, index_to_label, model_name,
                            AND=True):
    basepath = "/data4/dheeraj/metaguide/"
    dataset = "dblp/"
    # glove_dir = basepath + "glove.6B"
    dump_dir = basepath + "models/" + dataset + model_name + "/"
    tmp_dir = basepath + "checkpoints/" + dataset + model_name + "/"
    os.makedirs(dump_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    max_sentence_length = 100
    max_sentences = 15
    max_words = 20000
    embedding_dim = 100
    tokenizer = pickle.load(open(basepath + dataset + "tokenizer.pkl", "rb"))

    X, y, y_true, weights = get_weighted_train_data(df, labels, label_term_dict, label_author_dict, tokenizer,
                                                    label_to_index, AND=AND)
    print("****************** CLASSIFICATION REPORT FOR TRAINING DATA ********************")
    # df_train = create_training_df(X, y, y_true)
    # df_train.to_csv(basepath + dataset + "training_label.csv")
    y_vec = make_one_hot(y, label_to_index)
    print(classification_report(y_true, y))
    # y = np.array(y)
    # print("Fitting tokenizer...")
    # tokenizer = fit_get_tokenizer(X, max_words)
    print("Getting tokenizer")
    tokenizer = pickle.load(open(basepath + dataset + "tokenizer.pkl", "rb"))

    print("Splitting into train, dev...")
    X_train, y_train, X_val, y_val, weights_train, _ = create_train_dev_weights(X, labels=y_vec,
                                                                                weights=weights,
                                                                                tokenizer=tokenizer,
                                                                                max_sentences=max_sentences,
                                                                                max_sentence_length=max_sentence_length,
                                                                                max_words=max_words)
    # print("Creating Embedding matrix...")
    # embedding_matrix = create_embedding_matrix(glove_dir, tokenizer, embedding_dim)
    print("Getting Embedding matrix...")
    embedding_matrix = pickle.load(open(basepath + dataset + "embedding_matrix.pkl", "rb"))
    print("Initializing model...")
    model = HAN(max_words=max_sentence_length, max_sentences=max_sentences, output_size=len(y_train[0]),
                embedding_matrix=embedding_matrix)
    print("Compiling model...")
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
    print("model fitting - Hierachical attention network...")
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    mc = ModelCheckpoint(filepath=tmp_dir + 'model.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_acc', mode='max',
                         verbose=1, save_weights_only=True, save_best_only=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), nb_epoch=100, batch_size=256, callbacks=[es, mc],
              sample_weight=np.array(weights_train))
    # print("****************** CLASSIFICATION REPORT FOR DOCUMENTS WITH LABEL WORDS ********************")
    # X_label_all = prep_data(texts=X, max_sentences=max_sentences, max_sentence_length=max_sentence_length,
    #                         tokenizer=tokenizer)
    # pred = model.predict(X_label_all)
    # pred_labels = get_from_one_hot(pred, index_to_label)
    # print(classification_report(y_true, pred_labels))
    print("****************** CLASSIFICATION REPORT FOR All DOCUMENTS ********************")
    X_all = prep_data(texts=df["abstract"], max_sentences=max_sentences, max_sentence_length=max_sentence_length,
                      tokenizer=tokenizer)
    y_true_all = df["label"]
    pred = model.predict(X_all)
    pred_labels = get_from_one_hot(pred, index_to_label)
    print(classification_report(y_true_all, pred_labels))
    print("Dumping the model...")
    model.save_weights(dump_dir + "model_weights_" + model_name + ".h5")
    model.save(dump_dir + "model_" + model_name + ".h5")
    return pred_labels, pred


def train_classifier(df, labels, label_term_dict, label_author_dict, label_to_index, index_to_label,
                     model_name, old=True, soft=False):
    basepath = "/data4/dheeraj/covid-analysis/"
    dataset = ""
    # glove_dir = basepath + "glove.6B"
    dump_dir = basepath + "models/" + dataset + model_name + "/"
    tmp_dir = basepath + "checkpoints/" + dataset + model_name + "/"
    os.makedirs(dump_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    max_sentence_length = 100
    max_sentences = 15
    max_words = 20000
    embedding_dim = 100
    tokenizer = pickle.load(open(basepath + dataset + "tokenizer.pkl", "rb"))

    if old:
        X, y = get_train_data(df, labels, label_term_dict, label_author_dict, tokenizer, label_to_index,
                              ignore_metadata=False, soft=soft)
    else:
        X, y = get_confident_train_data(df, labels, label_term_dict, label_author_dict, label_to_index, tokenizer)
    print("****************** CLASSIFICATION REPORT FOR TRAINING DATA ********************")
    # df_train = create_training_df(X, y, y_true)
    # df_train.to_csv(basepath + dataset + "training_label.csv")
    if not soft:
        y_vec = make_one_hot(y, label_to_index)
    else:
        y_vec = np.array(y)
        y_argmax = np.argmax(y, axis=-1)
        y_str = []
        for i in y_argmax:
            y_str.append(index_to_label[i])
    # y = np.array(y)
    # print("Fitting tokenizer...")
    # tokenizer = fit_get_tokenizer(X, max_words)
    print("Getting tokenizer")
    tokenizer = pickle.load(open(basepath + dataset + "tokenizer.pkl", "rb"))

    print("Splitting into train, dev...")
    X_train, y_train, X_val, y_val, _, _ = create_train_dev(X, labels=y_vec, tokenizer=tokenizer,
                                                            max_sentences=max_sentences,
                                                            max_sentence_length=max_sentence_length,
                                                            max_words=max_words, val=False)
    # print("Creating Embedding matrix...")
    # embedding_matrix = create_embedding_matrix(glove_dir, tokenizer, embedding_dim)
    print("Getting Embedding matrix...")
    embedding_matrix = pickle.load(open(basepath + dataset + "embedding_matrix.pkl", "rb"))
    print("Initializing model...")
    model = HAN(max_words=max_sentence_length, max_sentences=max_sentences, output_size=len(y_train[0]),
                embedding_matrix=embedding_matrix)
    print("Compiling model...")
    model.summary()
    if not soft:
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['acc'])
    else:
        model.compile(loss=kullback_leibler_divergence, optimizer='adam', metrics=['acc'])
    print("model fitting - Hierachical attention network...")
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    mc = ModelCheckpoint(filepath=tmp_dir + 'model.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_acc', mode='max',
                         verbose=1, save_weights_only=True, save_best_only=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), nb_epoch=100, batch_size=256, callbacks=[es, mc])
    X_all = prep_data(texts=df["abstract"], max_sentences=max_sentences, max_sentence_length=max_sentence_length,
                      tokenizer=tokenizer)
    pred = model.predict(X_all)
    pred_labels = get_from_one_hot(pred, index_to_label)
    print("Dumping the model...")
    model.save_weights(dump_dir + "model_weights_" + model_name + ".h5")
    model.save(dump_dir + "model_" + model_name + ".h5")
    return pred_labels, pred
