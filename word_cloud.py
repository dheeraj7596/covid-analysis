import pickle
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

if __name__ == "__main__":
    base_path = "./data/"
    df = pickle.load(open(base_path + "df_29k.pkl", "rb"))
    abstracts = list(df.abstract)
    total_abstract = " ".join(abstracts)

    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(total_abstract)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    pass
