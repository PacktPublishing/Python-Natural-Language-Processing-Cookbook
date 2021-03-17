import nltk
import numpy as np
import string
import pickle
from gsdmm import MovieGroupProcess
from Chapter03.phrases import get_yelp_reviews
from Chapter04.preprocess_bbc_dataset import get_stopwords

tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
yelp_reviews_file = "Chapter03/yelp-dataset/review.json"
stopwords_file_path = "Chapter06/reviews_stopwords.csv"
stopwords = get_stopwords(stopwords_file_path)

def preprocess(text):
    sentences = tokenizer.tokenize(text)
    sentences = [nltk.tokenize.word_tokenize(sentence) for sentence in sentences]
    sentences = [list(set(word_list)) for word_list in sentences]
    sentences = [[word for word in word_list if word not in stopwords and word not in string.punctuation] for word_list in sentences]
    return sentences

def top_words_by_cluster(mgp, top_clusters, num_words):
    for cluster in top_clusters:
        sort_dicts = sorted(mgp.cluster_word_distribution[cluster].items(), key=lambda k: k[1], reverse=True)[:num_words]
        print(f'Cluster {cluster}: {sort_dicts}')

def main():
    reviews = get_yelp_reviews(yelp_reviews_file)
    sentences = preprocess(reviews)
    vocab = set(word for sentence in sentences for word in sentence)
    n_terms = len(vocab)
    mgp = MovieGroupProcess(K=25, alpha=0.1, beta=0.1, n_iters=30)
    mgp.fit(sentences, n_terms)
    pickle.dump(mgp, open("Chapter06/mgp.pkl", "wb"))
    doc_count = np.array(mgp.cluster_doc_count)
    print(doc_count)
    top_clusters = doc_count.argsort()[-15:][::-1]
    print(top_clusters)
    top_words_by_cluster(mgp, top_clusters, 10)

if (__name__ == "__main__"):
    main()

