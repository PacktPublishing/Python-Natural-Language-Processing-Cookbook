#import gensim
from gensim.models import KeyedVectors
import numpy as np

w2vec_model_path = "Chapter03/40/model.bin"

def load_model(path):
    model = KeyedVectors.load_word2vec_format(w2vec_model_path, binary=True)
    return model

def get_sentence_vector(word_vectors):
    matrix = np.array(word_vectors)
    centroid = np.mean(matrix[:,:], axis=0)
    return centroid

def get_word_vectors(sentence, model):
    word_vectors = []
    for word in sentence:
        try:
            word_vector = model.get_vector(word.lower())
            word_vectors.append(word_vector)
        except KeyError:
            continue
    return word_vectors

#print(wv_from_bin.vocab)

def main():
    model = load_model(w2vec_model_path)
    print(model['holmes'])
    print(model.most_similar(['holmes'], topn=15))
    sentence = "It was not that he felt any emotion akin to love for Irene Adler."
    word_vectors = get_word_vectors(sentence, model)
    sentence_vector = get_sentence_vector(word_vectors)
    words = ['banana', 'apple', 'computer', 'strawberry']
    print(model.doesnt_match(words))
    word = "cup"
    words = ['glass', 'computer', 'pencil', 'watch']
    print(model.most_similar_to_given(word, words))


if (__name__ == "__main__"):
    main()

