import nltk
from nltk.tokenize import TweetTokenizer
import time
import spacy

nlp = spacy.load("en_core_web_sm")


def read_text_file(filename):
    file = open(filename, "r", encoding="utf-8")
    return file.read()


def preprocess_text(text):
    text = text.replace("\n", " ")
    return text


def tokenize_nltk(text):
    return nltk.tokenize.word_tokenize(text)


def tokenize_spacy(text):
    doc = nlp(text)
    return [token.text for token in doc]


def tokenize(text):
    return tokenize_spacy(text)


def main():
    sherlock_holmes_text = read_text_file("Chapter01/sherlock_holmes_1.txt")
    sherlock_holmes_text = preprocess_text(sherlock_holmes_text)
    words = tokenize(sherlock_holmes_text)
    print(words)
    tweet = "@EmpireStateBldg Central Park Tower is reaaaaally hiiiiiiigh"
    words = nltk.tokenize.casual.casual_tokenize(
        tweet, preserve_case=True, reduce_len=True, strip_handles=True
    )
    print(words)


if __name__ == "__main__":
    start = time.time()
    main()
    print("%s s" % (time.time() - start))
