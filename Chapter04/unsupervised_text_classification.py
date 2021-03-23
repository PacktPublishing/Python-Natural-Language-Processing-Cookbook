import nltk
import re
import string
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.probability import FreqDist
from Chapter01.tokenization import tokenize_nltk
from Chapter01.dividing_into_sentences import divide_into_sentences_nltk
from Chapter04.preprocess_bbc_dataset import get_data
from Chapter04.keyword_classification import divide_data
from Chapter04.preprocess_bbc_dataset import get_stopwords

bbc_dataset = "Chapter04/bbc-text.csv"
stopwords_file_path = "Chapter01/stopwords.csv"
stopwords = get_stopwords(stopwords_file_path)
stemmer = SnowballStemmer('english')

def tokenize_and_stem(sentence):
    tokens = nltk.word_tokenize(sentence)
    filtered_tokens = [t for t in tokens if t not in stopwords and t not in string.punctuation and re.search('[a-zA-Z]', t)]
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def create_vectorizer(data):
    vec = TfidfVectorizer(max_df=0.90, min_df=0.05, stop_words=stopwords,
                          use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
    vec.fit(data)
    return vec

def make_predictions(test_data, vectorizer, km):
    predicted_data = {}
    for topic in test_data.keys():
        this_topic_list = test_data[topic]
        if (topic not in predicted_data.keys()):
            predicted_data[topic] = {}
        for text in this_topic_list:
            prediction = km.predict(vectorizer.transform([text]))[0]
            if (prediction not in predicted_data[topic].keys()):
                predicted_data[topic][prediction] = []
            predicted_data[topic][prediction].append(text)
    return predicted_data

def print_report(predicted_data):
    for topic in predicted_data.keys():
        print(topic)
        for prediction in predicted_data[topic].keys():
            print("Cluster number: ", prediction, "number of items: ", len(predicted_data[topic][prediction]))

def get_most_frequent_words(text):
    word_list = tokenize_nltk(text)
    word_list = [word for word in word_list if word not in stopwords and word not in string.punctuation and re.search('[a-zA-Z]', word)]
    freq_dist = FreqDist(word_list)
    top_200 = freq_dist.most_common(200)
    top_200 = [word[0] for word in top_200]
    return top_200

def print_most_common_words_by_cluster(all_training, km, num_clusters):
    clusters = km.labels_.tolist()
    docs = {'text': all_training, 'cluster': clusters}
    frame = pd.DataFrame(docs, index = [clusters])
    for cluster in range(0, num_clusters):
        this_cluster_text = frame[frame['cluster'] == cluster]
        all_text = " ".join(this_cluster_text['text'].astype(str))
        top_200 = get_most_frequent_words(all_text)
        print(cluster)
        print(top_200)
    return frame

def main():
    data_dict = get_data(bbc_dataset)
    (train_dict, test_dict) = divide_data(data_dict)
    all_training = []
    all_test = []
    for topic in train_dict.keys():
        all_training = all_training + train_dict[topic]
    for topic in test_dict.keys():
        all_test = all_test + test_dict[topic]
    vectorizer = create_vectorizer(all_training)
    matrix = vectorizer.transform(all_training)
    km = KMeans(n_clusters=5, init='k-means++', random_state=0)
    km.fit(matrix)
    predicted_data = make_predictions(test_dict, vectorizer, km)
    print_report(predicted_data)
    print_most_common_words_by_cluster(all_training, km, 5)
    pickle.dump(km, open("Chapter04/bbc_kmeans.pkl", "wb"))

if (__name__ == "__main__"):
    main()


