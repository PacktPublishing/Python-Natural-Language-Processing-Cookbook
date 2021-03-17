from sklearn.feature_extraction.text import CountVectorizer
from Chapter01.dividing_into_sentences import read_text_file, preprocess_text, divide_into_sentences_nltk

#https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction - Bigram vectorizer

def create_vectorizer(sentences):
    vectorizer = CountVectorizer(max_df=0.6)
    X = vectorizer.fit_transform(sentences)
    return (vectorizer, X)    

def get_sentences(filename):
    sherlock_holmes_text = read_text_file(filename)
    sherlock_holmes_text = preprocess_text(sherlock_holmes_text)
    sentences = divide_into_sentences_nltk(sherlock_holmes_text)
    return sentences

def get_new_sentence_vector(sentence, vectorizer):
    new_sentence_vector = vectorizer.transform([sentence])
    return new_sentence_vector

def create_bigram_vectorizer(sentences):
    bigram_vectorizer = CountVectorizer(ngram_range=(1, 2))
    X = bigram_vectorizer.fit_transform(sentences)
    return (bigram_vectorizer, X)

def main():
    sentences = get_sentences("Chapter01/sherlock_holmes_1.txt")
    (vectorizer, X) = create_vectorizer(sentences)
    print(X)
    print(type(X))
    print(X.todense())
    print(type(X.todense()))
    print(vectorizer.get_feature_names())


    new_sentence = "And yet there was but one woman to him, and that woman was the late Irene Adler, of dubious and questionable memory."
    new_sentence_vector = get_new_sentence_vector(new_sentence, vectorizer)

    analyze = vectorizer.build_analyzer()
    print(analyze(new_sentence))

    print(new_sentence_vector)
    print(new_sentence_vector.todense())

if (__name__ == "__main__"):
    main()
