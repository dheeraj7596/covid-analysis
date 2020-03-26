import pickle
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_path = "./data/"
    phrase_freq_map = pickle.load(open(data_path + "phrase_freq_map.pkl", "rb"))

    multi_word_phrase_freq_map = {}
    for p in phrase_freq_map:
        if len(p.strip().split()) > 1:
            multi_word_phrase_freq_map[p] = phrase_freq_map[p]

    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate_from_frequencies(
        multi_word_phrase_freq_map)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    pass
