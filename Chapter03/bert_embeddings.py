from sentence_transformers import SentenceTransformer
from Chapter01.dividing_into_sentences import read_text_file, divide_into_sentences_nltk


def main():
    text = read_text_file("Chapter01/sherlock_holmes.txt")
    sentences = divide_into_sentences_nltk(text)
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    sentence_embeddings = model.encode(["the beautiful lake"])

    print("Sentence embeddings:")
    print(sentence_embeddings)

if (__name__ == "__main__"):
    main()

#Installation
#conda create -n newenv python=3.6.10 anaconda
#conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
#pip install transformers
#pip install -U sentence-transformers
