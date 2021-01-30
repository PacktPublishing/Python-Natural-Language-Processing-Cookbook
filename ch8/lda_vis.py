import gensim
import pyLDAvis.gensim

dictionary = gensim.corpora.Dictionary.load('ch6/gensim/id2word.dict')
corpus = gensim.corpora.MmCorpus('ch6/gensim/corpus.mm')
lda = gensim.models.ldamodel.LdaModel.load('ch6/gensim/lda_gensim.model')

lda_prepared = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
pyLDAvis.show(lda_prepared)
pyLDAvis.save_html(lda_prepared, 'ch8/lda.html')