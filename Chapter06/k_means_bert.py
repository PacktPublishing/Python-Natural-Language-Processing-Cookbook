import re
import string
import pandas as pd
from sklearn.cluster import KMeans
from nltk.probability import FreqDist
from Chapter01.tokenization import tokenize_nltk
from Chapter04.preprocess_bbc_dataset import get_data
from Chapter04.keyword_classification import divide_data
from Chapter04.preprocess_bbc_dataset import get_stopwords
from Chapter04.unsupervised_text_classification import get_most_frequent_words, print_most_common_words_by_cluster
from Chapter06.lda_topic_sklearn import stopwords, bbc_dataset, new_example
from Chapter06.lda_topic_gensim import preprocess
from sentence_transformers import SentenceTransformer

bbc_dataset = "Chapter04/bbc-text.csv"
stopwords_file_path = "Chapter01/stopwords.csv"
stopwords = get_stopwords(stopwords_file_path)

def test_new_example(km, model, example):
    embedded = model.encode([example])
    topic = km.predict(embedded)[0]
    print(topic)
    return topic

def main():
    df = pd.read_csv(bbc_dataset)
    df = preprocess(df)
    df['text'] = df['text'].apply(lambda x: " ".join(x))
    documents = df['text'].values
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    encoded_data = model.encode(documents)
    km = KMeans(n_clusters=5, random_state=0)
    km.fit(encoded_data)
    print_most_common_words_by_cluster(documents, km, 5)
    test_new_example(km, model, new_example)

if (__name__ == "__main__"):
    main()

