import pickle
from nltk.corpus import stopwords

if __name__ == "__main__":
    basepath = "./data/"
    pkl_dump_dir = basepath

    df = pickle.load(open(pkl_dump_dir + "df_29k_abstract_phrase.pkl", "rb"))
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')

    abstracts = list(df["abstract"])

    clean_abstracts = []
    for abs in abstracts:
        word_list = abs.strip().split()
        filtered_words = [word for word in word_list if word not in stop_words]
        clean_abstracts.append(" ".join(filtered_words))

    df["abstract"] = clean_abstracts

    pickle.dump(df, open(pkl_dump_dir + "df_29k_abstract_phrase_removed_stopwords.pkl", "wb"))
