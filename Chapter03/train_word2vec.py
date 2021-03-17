import gensim
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
import pickle
from os import listdir
from os.path import isfile, join
from Chapter03.bag_of_words import get_sentences
from Chapter01.tokenization import tokenize_nltk

word2vec_model_path = "word2vec.model"
books_dir = "1025_1853_bundle_archive"
evaluation_file = "Chapter03/questions-words.txt"
pretrained_model_path = "Chapter03/40/model.bin"

def train_word2vec(words, word2vec_model_path):
    #model = gensim.models.Word2Vec(
    #    words,
    #    size=50,
    #    window=7,
    #    min_count=1,
    #    workers=10)
    model = gensim.models.Word2Vec(words, window=5, size=200, min_count=5)
    model.train(words, total_examples=len(words), epochs=200)
    pickle.dump(model, open(word2vec_model_path, 'wb'))
    return model

def get_all_book_sentences(directory):
    text_files = [join(directory, f) for f in listdir(directory) if isfile(join(directory, f)) and ".txt" in f]
    all_sentences = []
    for text_file in text_files:
        sentences = get_sentences(text_file)
        all_sentences = all_sentences + sentences
    return all_sentences

def test_model():
    model = pickle.load(open(word2vec_model_path, 'rb'))
    words = list(model.wv.vocab)
    print(words)
    w1 = "river"
    words = model.wv.most_similar(w1, topn=10)
    print(words)

def evaluate_model(model, filename):
    return model.wv.accuracy(filename)

def main():
    sentences = get_all_book_sentences(books_dir)
    sentences = [tokenize_nltk(s.lower()) for s in sentences]
    #model = train_word2vec(sentences)
    #test_model()
    model = pickle.load(open(word2vec_model_path, 'rb'))
    #accuracy_list = evaluate_model(model, evaluation_file)
    #print(accuracy_list)
    (analogy_score, word_list) = model.wv.evaluate_word_analogies(datapath('questions-words.txt'))
    print(analogy_score)
    pretrained_model = KeyedVectors.load_word2vec_format(pretrained_model_path, binary=True)
    (analogy_score, word_list) = pretrained_model.evaluate_word_analogies(datapath('questions-words.txt'))
    print(analogy_score)

if (__name__ == "__main__"):
    main()