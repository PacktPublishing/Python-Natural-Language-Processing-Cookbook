import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from Chapter01.removing_stopwords import read_in_csv
from Chapter03.bag_of_words import get_sentences

stemmer = SnowballStemmer('english')
stopwords_file_path = "Chapter01/stopwords.csv"

def tokenize_and_stem(sentence):
    tokens = nltk.word_tokenize(sentence)
    filtered_tokens = [t for t in tokens if t not in string.punctuation]
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def create_char_vectorizer(sentences):
    #Create TF-IDF object
    tfidf_char_vectorizer = TfidfVectorizer(analyzer='char_wb', max_df=0.90, max_features=200000,
                                        min_df=0.05, use_idf=True, ngram_range=(1,3))
    tfidf_char_vectorizer = tfidf_char_vectorizer.fit(sentences)
    tfidf_matrix = tfidf_char_vectorizer.transform(sentences)
    print(tfidf_matrix)
    dense_matrix = tfidf_matrix.todense()
    print(dense_matrix)
    print(tfidf_char_vectorizer.get_feature_names())
    analyze = tfidf_char_vectorizer.build_analyzer()
    print(analyze("To Sherlock Holmes she is always _the_ woman."))
    return (tfidf_char_vectorizer, tfidf_matrix)    

def create_vectorizer(sentences):
    #Create TF-IDF object
    stopword_list = read_in_csv(stopwords_file_path)
    stemmed_stopwords = [tokenize_and_stem(stopword)[0] for stopword in stopword_list]
    stopword_list = stopword_list + stemmed_stopwords
    tfidf_vectorizer = TfidfVectorizer(max_df=0.90, max_features=200000,
                                        min_df=0.05, stop_words=stopword_list,
                                        use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
    tfidf_vectorizer = tfidf_vectorizer.fit(sentences)
    tfidf_matrix = tfidf_vectorizer.transform(sentences)
    print(tfidf_matrix)
    dense_matrix = tfidf_matrix.todense()
    print(dense_matrix)
    print(tfidf_vectorizer.get_feature_names())
    analyze = tfidf_vectorizer.build_analyzer()
    print(analyze("To Sherlock Holmes she is always _the_ woman."))
    return (tfidf_vectorizer, tfidf_matrix)

def main():
    sentences = get_sentences("Chapter01/sherlock_holmes_1.txt")
    (vectorizer, matrix) = create_vectorizer(sentences)


if (__name__ == "__main__"):
    main()
    