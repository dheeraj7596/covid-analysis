from bs4 import BeautifulSoup
import bleach
import pickle
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt


def generate_name(id):
    return "fnust" + str(id)


def decrypt(val):
    try:
        if val[:5] == "fnust":
            return int(val[5:])
    except:
        return None


if __name__ == "__main__":
    base_path = "./data/"
    data_path = base_path
    out_path = data_path + "segmentation.txt"
    df = pickle.load(open(data_path + "df_29k.pkl", "rb"))
    f = open(out_path, "r")
    lines = f.readlines()
    f.close()

    phrase_freq_map = {}
    counter = 0
    data = []

    for line in lines:
        line = line.lower()
        soup = BeautifulSoup(line)
        for p in soup.findAll("phrase"):
            phrase = p.string
            try:
                phrase_freq_map[phrase] += 1
            except:
                phrase_freq_map[phrase] = 1

    phrase_freq_map = {k: v for k, v in sorted(phrase_freq_map.items(), key=lambda item: -item[1])}
    pickle.dump(phrase_freq_map, open(base_path + "phrase_freq_map.pkl", "wb"))

    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate_from_frequencies(phrase_freq_map)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    pass
