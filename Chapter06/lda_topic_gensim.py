import re
import pandas as pd
from gensim.models.ldamodel import LdaModel
import gensim.corpora as corpora
from gensim.corpora import MmCorpus
from gensim.utils import simple_preprocess
from pprint import pprint
from Chapter06.lda_topic_sklearn import stopwords, bbc_dataset, clean_data, new_example

model_path = "Chapter06/gensim/lda_gensim.model"
dict_path = "Chapter06/gensim/id2word.dict"
corpus_path = "Chapter06/gensim/corpus.mm"


def preprocess(df):
    df = clean_data(df)
    df['text'] = df['text'].apply(lambda x: simple_preprocess(x))
    df['text'] = df['text'].apply(lambda x: [word for word in x if word not in stopwords])
    return df

def create_lda_model(id_dict, corpus, num_topics):
    lda_model = LdaModel(corpus=corpus,
                         id2word=id_dict,
                         num_topics=num_topics, 
                         random_state=100,
                         chunksize=100,
                         passes=10)
    return lda_model

def save_model(lda, lda_path, id_dict, dict_path, corpus, corpus_path):
    lda.save(lda_path)
    id_dict.save(dict_path)
    MmCorpus.serialize(corpus_path, corpus)

def load_model(lda_path, dict_path):
    lda = LdaModel.load(lda_path)
    id_dict = corpora.Dictionary.load(dict_path)
    return (lda, id_dict)

def clean_text(input_string):
    input_string = re.sub(r'[^\w\s]', ' ', input_string)
    input_string = re.sub(r'\d', '', input_string)
    input_list = simple_preprocess(input_string)
    input_list = [word for word in input_list if word not in stopwords]
    return input_list

def test_new_example(model, id_dict, input_string):
    input_list = clean_text(input_string)
    bow = id_dict.doc2bow(input_list)
    topics = model[bow]
    print(topics)

def load_model_test_new_example():
    (lda_model, id_dict) = load_model(model_path, dict_path)
    test_new_example(lda_model, id_dict, new_example)

def main():
    df = pd.read_csv(bbc_dataset)
    df = preprocess(df)
    texts = df['text'].values
    id_dict = corpora.Dictionary(texts)
    corpus = [id_dict.doc2bow(text) for text in texts]
    number_topics = 5
    lda_model = create_lda_model(id_dict, corpus, number_topics)
    pprint(lda_model.print_topics())
    #plot_perplexity(id_dict, corpus)
    save_model(lda_model, model_path, id_dict, dict_path, corpus, corpus_path)

if (__name__ == "__main__"):
    main()
    #load_model_test_new_example()


