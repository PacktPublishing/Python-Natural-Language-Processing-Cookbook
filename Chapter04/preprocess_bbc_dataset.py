import csv
import nltk
import string
import re
import math
import numpy as np
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn import preprocessing
from nltk.stem.snowball import SnowballStemmer
from Chapter01.tokenization import tokenize_nltk

stemmer = SnowballStemmer('english')
bbc_dataset = "Chapter04/bbc-text.csv"
stopwords_file_path = "Chapter01/stopwords.csv"
stopwords = []

def read_in_csv(csv_file):
    with open(csv_file, 'r', encoding='utf-8') as fp:
        reader = csv.reader(fp, delimiter=',', quotechar='"')
        data_read = [row for row in reader]
    return data_read

def tokenize_and_stem(sentence):
    tokens = nltk.word_tokenize(sentence)
    filtered_tokens = [t for t in tokens if t not in string.punctuation]
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def get_stopwords(path=stopwords_file_path):
    stopwords = read_in_csv(path)
    stopwords = [word[0] for word in stopwords]
    stemmed_stopwords = [stemmer.stem(word) for word in stopwords]
    stopwords = stopwords + stemmed_stopwords
    return stopwords

stopwords = get_stopwords(stopwords_file_path)

def get_data(filename):
    data = read_in_csv(filename)
    data_dict = {}
    for row in data[1:]:
        category = row[0]
        text = row[1]
        if (category not in data_dict.keys()):
            data_dict[category] = []
        data_dict[category].append(text)
    return data_dict    

def get_stats(text, num_words=200):
    word_list = tokenize_nltk(text)
    word_list = [word for word in word_list if word not in stopwords and re.search("[A-Za-z]", word)]
    freq_dist = FreqDist(word_list)
    print(freq_dist.most_common(num_words))
    return freq_dist

def predict_trivial(X_train, y_train, X_test, y_test, le):
    dummy_clf = DummyClassifier(strategy='uniform', random_state=0)
    dummy_clf.fit(X_train, y_train)
    y_pred = dummy_clf.predict(X_test)
    print(dummy_clf.score(X_test, y_test))
    print(classification_report(y_test, y_pred, labels=le.transform(le.classes_), target_names=le.classes_))

def create_vectorizer(text_list):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=0.05, stop_words='english',
                                       use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
    tfidf_vectorizer.fit_transform(text_list)
    return tfidf_vectorizer

def get_labels(names):
    le = preprocessing.LabelEncoder()
    le.fit(names)
    return le

def create_data_matrix(input_data, vectorizer, label, le):
    vectors = vectorizer.transform(input_data).todense()
    labels = [label]*len(input_data)
    enc_labels = le.transform(labels)
    return (vectors, enc_labels)

def create_dataset(vectorizer, data_dict, le):
    business_news = data_dict["business"]
    sports_news = data_dict["sport"]
    (sports_vectors, sports_labels) = create_data_matrix(sports_news, vectorizer, "sport", le)
    (business_vectors, business_labels) = create_data_matrix(business_news, vectorizer, "business", le)
    all_data_matrix = np.vstack((business_vectors, sports_vectors))
    labels = np.concatenate([business_labels, sports_labels])
    return (all_data_matrix, labels)


def split_test_train(data, train_percent):
    train_test_border = math.ceil(train_percent*len(data))
    train_data = data[0:train_test_border]
    test_data = data[train_test_border:]
    return (train_data, test_data)

def main():
    data_dict = get_data(bbc_dataset)
    for topic in data_dict.keys():
        print(topic, "\t", len(data_dict[topic]))
    business_data = data_dict["business"]
    sports_data = data_dict["sport"]
    business_string = " ".join(business_data)
    sports_string = " ".join(sports_data)
    get_stats(business_string)
    get_stats(sports_string)

    (business_train_data, business_test_data) = split_test_train(business_data, 0.8)
    (sports_train_data, sports_test_data) = split_test_train(sports_data, 0.8)
    train_data = business_train_data + sports_train_data
    tfidf_vec = create_vectorizer(train_data)
    le = get_labels(["business", "sport"])
    
    train_data_dict = {'business':business_train_data, 'sport':sports_train_data}
    test_data_dict = {'business':business_test_data, 'sport':sports_test_data}
    (X_train, y_train) = create_dataset(tfidf_vec, train_data_dict, le)
    (X_test, y_test) = create_dataset(tfidf_vec, test_data_dict, le)
    predict_trivial(X_train, y_train, X_test, y_test, le)

if (__name__ == "__main__"):
    main()

