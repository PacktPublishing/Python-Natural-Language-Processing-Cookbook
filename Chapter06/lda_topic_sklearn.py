import re
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Chapter04.preprocess_bbc_dataset import get_stopwords
from Chapter04.unsupervised_text_classification import tokenize_and_stem

stopwords_file_path = "Chapter01/stopwords.csv"
stopwords = get_stopwords(stopwords_file_path)

bbc_dataset = "Chapter04/bbc-text.csv"
model_path = "Chapter06/sklearn/lda_sklearn.pkl"
vectorizer_path = "Chapter06/sklearn/vectorizer.pkl"

new_example = """Manchester United players slumped to the turf 
at full-time in Germany on Tuesday in acknowledgement of what their 
latest pedestrian first-half display had cost them. The 3-2 loss at 
RB Leipzig means United will not be one of the 16 teams in the draw 
for the knockout stages of the Champions League. And this is not the 
only price for failure. The damage will be felt in the accounts, in 
the dealings they have with current and potentially future players 
and in the faith the fans have placed in manager Ole Gunnar Solskjaer. 
With Paul Pogba's agent angling for a move for his client and ex-United 
defender Phil Neville speaking of a "witchhunt" against his former team-mate 
Solskjaer, BBC Sport looks at the ramifications and reaction to a big loss for United."""

def create_tf_idf_vectorizer(documents):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, max_features=20000,
                                        stop_words=stopwords, tokenizer=tokenize_and_stem, 
                                        use_idf=True)
    data = tfidf_vectorizer.fit_transform(documents)
    return (tfidf_vectorizer, data)

def create_count_vectorizer(documents):
    count_vectorizer = CountVectorizer(stop_words=stopwords, tokenizer=tokenize_and_stem, max_features=1500)
    data = count_vectorizer.fit_transform(documents)
    return (count_vectorizer, data)

def create_and_fit_lda(data, num_topics):
    lda = LDA(n_components=num_topics, n_jobs=-1)
    lda.fit(data)
    return lda

def clean_data(df):
    df['text'] = df['text'].apply(lambda x: re.sub(r'[^\w\s]', ' ', x))
    df['text'] = df['text'].apply(lambda x: re.sub(r'\d', '', x))
    return df

def get_most_common_words_for_topics(model, vectorizer, n_top_words):
    words = vectorizer.get_feature_names()
    word_dict = {}
    for topic_index, topic in enumerate(model.components_):
        this_topic_words = [words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        word_dict[topic_index] = this_topic_words
    return word_dict

def print_topic_words(word_dict):
    for key in word_dict.keys():
        print(f"Topic {key}")
        print("\t", word_dict[key])

def get_train_test_sets(df, vectorizer):
    train, test = train_test_split(list(df['text'].values), test_size = 0.2)
    train_data = vectorizer.fit_transform(train)
    test_data = vectorizer.fit_transform(test)
    return (train_data, test_data)

def save_model(lda, lda_path, vect, vect_path):
    pickle.dump(lda, open(lda_path, 'wb'))
    pickle.dump(vect, open(vect_path, 'wb'))

def test_new_example(lda, vect, example):
    vectorized = vect.transform([example])
    topic = lda.transform(vectorized)
    print(topic)
    return topic

def main():
    df = pd.read_csv(bbc_dataset)
    df = clean_data(df)
    documents = df['text']
    number_topics = 5
    (vectorizer, data) = create_count_vectorizer(documents)
    #(vectorizer, data) = create_tf_idf_vectorizer(documents)
    lda = create_and_fit_lda(data, number_topics)
    topic_words = get_most_common_words_for_topics(lda, vectorizer, 10)
    print_topic_words(topic_words)
    save_model(lda, model_path, vectorizer, vectorizer_path)
    test_new_example(lda, vectorizer, new_example)

if (__name__ == "__main__"):
    main()

