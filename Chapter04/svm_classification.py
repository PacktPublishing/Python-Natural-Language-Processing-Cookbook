import numpy as np
import pandas as pd
import string
import pickle
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from Chapter01.tokenization import tokenize_nltk
from Chapter04.unsupervised_text_classification import tokenize_and_stem
from Chapter04.preprocess_bbc_dataset import get_data
from Chapter04.keyword_classification import get_labels
from Chapter04.preprocess_bbc_dataset import get_stopwords

bbc_dataset = "Chapter04/bbc-text.csv"
stopwords_file_path = "Chapter01/stopwords.csv"
stopwords = get_stopwords(stopwords_file_path)

new_example = """iPhone 12: Apple makes jump to 5G
Apple has confirmed its iPhone 12 handsets will be its first to work on faster 5G networks. 
The company has also extended the range to include a new "Mini" model that has a smaller 5.4in screen. 
The US firm bucked a wider industry downturn by increasing its handset sales over the past year. 
But some experts say the new features give Apple its best opportunity for growth since 2014, when it revamped its line-up with the iPhone 6. 
"5G will bring a new level of performance for downloads and uploads, higher quality video streaming, more responsive gaming, 
real-time interactivity and so much more," said chief executive Tim Cook. 
There has also been a cosmetic refresh this time round, with the sides of the devices getting sharper, flatter edges. 
The higher-end iPhone 12 Pro models also get bigger screens than before and a new sensor to help with low-light photography. 
However, for the first time none of the devices will be bundled with headphones or a charger. 
Apple said the move was to help reduce its impact on the environment. "Tim Cook [has] the stage set for a super-cycle 5G product release," 
commented Dan Ives, an analyst at Wedbush Securities. 
He added that about 40% of the 950 million iPhones in use had not been upgraded in at least three-and-a-half years, presenting a "once-in-a-decade" opportunity. 
In theory, the Mini could dent Apple's earnings by encouraging the public to buy a product on which it makes a smaller profit than the other phones. 
But one expert thought that unlikely. 
"Apple successfully launched the iPhone SE in April by introducing it at a lower price point without cannibalising sales of the iPhone 11 series," noted Marta Pinto from IDC. 
"There are customers out there who want a smaller, cheaper phone, so this is a proven formula that takes into account market trends." 
The iPhone is already the bestselling smartphone brand in the UK and the second-most popular in the world in terms of market share. 
If forecasts of pent up demand are correct, it could prompt a battle between network operators, as customers become more likely to switch. 
"Networks are going to have to offer eye-wateringly attractive deals, and the way they're going to do that is on great tariffs and attractive trade-in deals," 
predicted Ben Wood from the consultancy CCS Insight. Apple typically unveils its new iPhones in September, but opted for a later date this year. 
It has not said why, but it was widely speculated to be related to disruption caused by the coronavirus pandemic. The firm's shares ended the day 2.7% lower. 
This has been linked to reports that several Chinese internet platforms opted not to carry the livestream, 
although it was still widely viewed and commented on via the social media network Sina Weibo."""

def create_dataset(data_dict, le):
    text = []
    labels = []
    for topic in data_dict:
        label = le.transform([topic])
        text = text + data_dict[topic]
        this_topic_labels = [label[0]]*len(data_dict[topic])
        labels = labels + this_topic_labels
    docs = {'text':text, 'label':labels}
    frame = pd.DataFrame(docs)
    return frame

def split_dataset(df, train_column_name, gold_column_name, test_percent):
    X_train, X_test, y_train, y_test = train_test_split(df[train_column_name], df[gold_column_name], test_size=test_percent, random_state=0)
    return (X_train, X_test, y_train, y_test)

def create_and_fit_vectorizer(training_text):
    vec = TfidfVectorizer(max_df=0.90, min_df=0.05, stop_words=stopwords,
                          use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
    return vec.fit(training_text)
    
def train_svm_classifier(X_train, y_train):
    clf = svm.SVC(C=1, kernel='linear')
    clf = clf.fit(X_train, y_train)
    return clf

def evaluate(clf, X_test, y_test, le):
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, labels=le.transform(le.classes_), target_names=le.classes_))

def test_new_example(input_string, clf, vectorizer, le):
    vector = vectorizer.transform([input_string]).todense()
    prediction = clf.predict(vector)
    print(prediction)
    label = le.inverse_transform(prediction)
    print(label)

def main():
    data_dict = get_data(bbc_dataset)
    le = get_labels(list(data_dict.keys()))
    df = create_dataset(data_dict, le)
    (X_train, X_test, y_train, y_test) = split_dataset(df, 'text', 'label', 0.2)
    vectorizer = create_and_fit_vectorizer(X_train)
    X_train = vectorizer.transform(X_train).todense()
    X_test = vectorizer.transform(X_test).todense()
    clf = train_svm_classifier(X_train, y_train)
    pickle.dump(clf, open("Chapter04/bbc_svm.pkl", "wb"))
    #clf = pickle.load(open("Chapter04/bbc_svm.pkl", "rb"))
    evaluate(clf, X_test, y_test, le)
    test_new_example(new_example, clf, vectorizer, le)


if (__name__ == "__main__"):
    main()


