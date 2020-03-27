import pickle

if __name__ == "__main__":
    base_path = "./data/"
    df = pickle.load(open(base_path + "df_29k_abstract.pkl", "rb"))
    abstracts = list(df.abstract)
    f = open(base_path + "abs.txt", "w")
    for abs in abstracts:
        if len(abs) > 0:
            f.write(abs)
            f.write("\n")
    f.close()
