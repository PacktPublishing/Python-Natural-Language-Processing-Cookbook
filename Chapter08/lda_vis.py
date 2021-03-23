import gensim
import pyLDAvis.gensim

dictionary = gensim.corpora.Dictionary.load('Chapter06/gensim/id2word.dict')
corpus = gensim.corpora.MmCorpus('Chapter06/gensim/corpus.mm')
lda = gensim.models.ldamodel.LdaModel.load('Chapter06/gensim/lda_gensim.model')

lda_prepared = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
pyLDAvis.show(lda_prepared)
pyLDAvis.save_html(lda_prepared, 'Chapter08/lda.html')