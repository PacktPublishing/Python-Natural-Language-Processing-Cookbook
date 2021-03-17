import numpy as np
import string
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from itertools import repeat
from nltk.probability import FreqDist
from Chapter01.tokenization import tokenize_nltk
from Chapter04.preprocess_bbc_dataset import get_data
from Chapter04.preprocess_bbc_dataset import get_stopwords

business_vocabulary = ["market", "company", "growth", "firm", "economy", "government", "bank", 
                       "sales", "oil", "prices", "business", "uk", "financial", "dollar", "stock", 
                       "trade", "investment", "quarter", "profit", "jobs", "foreign", "tax",
                       "euro", "budget", "cost", "money", "investor", "industry", "million", "debt"]

sports_vocabulary = ["game", "england", "win", "player", "cup", "team", "club", "match",
                     "set", "final", "coach", "season", "injury", "victory", "league", "play",
                     "champion", "olympic", "title", "ball", "sport", "race", "football", "rugby",
                     "tennis", "basketball", "hockey"]


business_vectorizer = CountVectorizer(vocabulary=business_vocabulary)
sports_vectorizer = CountVectorizer(vocabulary=sports_vocabulary)

bbc_dataset = "Chapter04/bbc-text.csv"
stopwords_file_path = "Chapter01/stopwords.csv"
stopwords = get_stopwords(stopwords_file_path)

def get_labels(labels):
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    return le

le = get_labels(["business", "sport"])

def create_vectorizers(data_dict):
    topic_list = list(data_dict.keys())
    vectorizer_dict = {}
    for topic in topic_list:
        text_array = data_dict[topic]
        text = " ".join(text_array)
        word_list = tokenize_nltk(text)
        word_list = [word for word in word_list if word not in stopwords]
        freq_dist = FreqDist(word_list)
        top_200 = freq_dist.most_common(200)
        vocab = [wtuple[0] for wtuple in top_200 if wtuple[0] not in stopwords and wtuple[0] not in string.punctuation]
        vectorizer_dict[topic] = CountVectorizer(vocabulary=vocab)
    return vectorizer_dict

def create_dataset(data_dict, le):
    data_matrix = []
    classifications = []
    gold_labels = []
    for text in data_dict["business"]:
        gold_labels.append(le.transform(["business"]))
        text_vector = transform(text)
        data_matrix.append(text_vector)
    for text in data_dict["sport"]:
        gold_labels.append(le.transform(["sport"]))
        text_vector = transform(text)
        data_matrix.append(text_vector)
    X = np.array(data_matrix)
    y = np.array(gold_labels)
    return (X, y)

def create_dataset_auto(data_dict, le, vectorizer_dict):
    data_matrix = []
    classifications = []
    gold_labels = []
    for topic in data_dict.keys():
        for text in data_dict[topic]:
            gold_labels.append(le.transform([topic]))
            text_vector = transform_auto(text, vectorizer_dict, le)
            data_matrix.append(text_vector)
    X = np.array(data_matrix)
    y = np.array(gold_labels)
    return (X, y)


def transform(text):
    business_X = business_vectorizer.transform([text])
    sports_X = sports_vectorizer.transform([text])
    business_sum = sum(business_X.todense().tolist()[0])
    sports_sum = sum(sports_X.todense().tolist()[0])
    return np.array([business_sum, sports_sum])

def transform_auto(text, vect_dict, le):
    number_topics = len(list(vect_dict.keys()))
    sum_list = [0]*number_topics
    for topic in vect_dict.keys():
        vectorizer = vect_dict[topic]
        this_topic_matrix = vectorizer.transform([text])
        this_topic_sum = sum(this_topic_matrix.todense().tolist()[0])
        index = le.transform([topic])[0]
        sum_list[index] = this_topic_sum
    return np.array(sum_list)


def evaluate(X, y):
    y_pred = np.array(list(map(classify, X, repeat(le))))
    print(classification_report(y, y_pred, labels=le.transform(le.classes_), target_names=le.classes_))

def evaluate_auto(X, y, le):
    y_pred = np.array(list(map(classify_auto, X, repeat(le))))
    print(classification_report(y, y_pred, labels=le.transform(le.classes_), target_names=le.classes_))

def classify(vector, le):
    label = ""
    if (vector[0] > vector[1]):
        label = "business"
    else:
        label = "sport"
    return le.transform([label])

def classify_auto(vector, le):
    result = np.where(vector == np.amax(vector))
    label = result[0][0]
    return [label]

def main():
    data_dict = get_data(bbc_dataset)
    (X, y) = create_dataset(data_dict, le)
    evaluate(X, y)

def auto():
    data_dict = get_data(bbc_dataset)
    (train_dict, test_dict) = divide_data(data_dict)
    le = get_labels(list(data_dict.keys()))
    vectorizers = create_vectorizers(train_dict)
    (X, y) = create_dataset_auto(test_dict, le, vectorizers)
    evaluate_auto(X, y, le)

def divide_data(data_dict):
    train_dict = {}
    test_dict = {}
    for topic in data_dict.keys():
        text_list = data_dict[topic]
        x_train, x_test = train_test_split(text_list, test_size=0.2)
        train_dict[topic] = x_train
        test_dict[topic] = x_test
    return (train_dict, test_dict)

if (__name__ == "__main__"):
    #auto()
    main()

