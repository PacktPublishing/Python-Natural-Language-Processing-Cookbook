import re
import pandas as pd
from gensim.models.nmf import Nmf
from gensim.models import CoherenceModel
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
import matplotlib.pyplot as plt
from pprint import pprint
from Chapter06.lda_topic_sklearn import stopwords, bbc_dataset, new_example
from Chapter06.lda_topic_gensim import preprocess, test_new_example


def create_nmf_model(id_dict, corpus, num_topics):
    nmf_model = Nmf(corpus=corpus,
                         id2word=id_dict,
                         num_topics=num_topics, 
                         random_state=100,
                         chunksize=100,
                         passes=50)
    return nmf_model

def plot_coherence(id_dict, corpus, texts):
    num_topics_range = range(2, 10)
    coherences = []
    for num_topics in num_topics_range:
        nmf_model = create_nmf_model(id_dict, corpus, num_topics)
        coherence_model_nmf = CoherenceModel(model=nmf_model, texts=texts, dictionary=id_dict, coherence='c_npmi')
        coherences.append(coherence_model_nmf.get_coherence())
    plt.plot(num_topics_range, coherences, color='blue', marker='o', markersize=5)
    plt.title('Coherence as a function of number of topics')
    plt.xlabel('Number of topics')
    plt.ylabel('Coherence')
    plt.grid()
    plt.show()

def main():
    df = pd.read_csv(bbc_dataset)
    df = preprocess(df)
    texts = df['text'].values
    id_dict = corpora.Dictionary(texts)
    corpus = [id_dict.doc2bow(text) for text in texts]
    number_topics = 5
    nmf_model = create_nmf_model(id_dict, corpus, number_topics)
    pprint(nmf_model.print_topics())
    test_new_example(nmf_model, id_dict, new_example)
    plot_coherence(id_dict, corpus, texts)

if (__name__ == "__main__"):
    main()