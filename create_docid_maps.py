from word2vec import create_corpus, create_vocabulary
from construct_graph import detect_phrase
import pickle


def create_phrase_doc_id_map(df, tokenizer, phrase_id_map):
    phrase_docid = {}

    index_word = {}
    for w in tokenizer.word_index:
        index_word[tokenizer.word_index[w]] = w

    id_phrase_map = {}
    for ph in phrase_id_map:
        id_phrase_map[phrase_id_map[ph]] = ph

    for i, row in df.iterrows():
        abstract_str = row["abstract"]
        phrases = detect_phrase(abstract_str, tokenizer, index_word, id_phrase_map, i)
        for ph in phrases:
            try:
                phrase_docid[ph].add(i)
            except:
                phrase_docid[ph] = {i}
    return phrase_docid


def create_author_doc_id_map(df):
    author_docid = {}
    for i, row in df.iterrows():
        auts = row["authors"]
        for author in auts:
            try:
                author_docid[author].add(i)
            except:
                author_docid[author] = {i}
    return author_docid


if __name__ == "__main__":
    data_path = "./data/"

    df = pickle.load(open(data_path + "df_29k_abstract_phrase_removed_stopwords.pkl", "rb"))
    tokenizer = pickle.load(open(data_path + "tokenizer.pkl", "rb"))
    phrase_id_map = pickle.load(open(data_path + "phrase_id_map.pkl", "rb"))

    phrase_docid = create_phrase_doc_id_map(df, tokenizer, phrase_id_map)
    author_docid_map = create_author_doc_id_map(df)

    pickle.dump(phrase_docid, open(data_path + "phrase_docid_map.pkl", "wb"))
    pickle.dump(author_docid_map, open(data_path + "author_docid_map.pkl", "wb"))
