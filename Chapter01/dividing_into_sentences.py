import time
import nltk
import spacy

tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
nlp = spacy.load("en_core_web_sm")

def read_text_file(filename):
    file = open(filename, "r", encoding="utf-8") 
    return file.read()

def preprocess_text(text):
    text = text.replace("\n", " ")
    return text

def divide_into_sentences_nltk(text):
    sentences = tokenizer.tokenize(text)
    return sentences

def divide_into_sentences_spacy(text):
    doc = nlp(text)
    return [sentence.text for sentence in doc.sents]

def divide_into_sentences(text):
    return divide_into_sentences_nltk(text)

def main():
    sherlock_holmes_text = read_text_file("Chapter01/sherlock_holmes_1.txt")
    sherlock_holmes_text = preprocess_text(sherlock_holmes_text)
    sentences = divide_into_sentences(sherlock_holmes_text)
    print(sentences)

if __name__ == '__main__':
    start = time.time()
    main()
    print("%s s" % (time.time() - start))
