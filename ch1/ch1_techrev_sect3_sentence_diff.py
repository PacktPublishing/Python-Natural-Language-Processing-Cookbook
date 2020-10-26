"""
All of the steps in the installation could be improved:
```shell
$ conda create -n nlp_book python=3.6.10
$ conda activate nlp_book

$ conda install -n nlp_book nltk==3.4.5
$ conda install -n nlp_book spacy==2.1.1

python -m spacy download en_core_web_sm
# python -m spacy download en_core_web_md
# python -m spacy download en_core_web_lg
"""


############################################
# NLTK Sentence Segmenter

# 1. Import the nltk package:
import nltk
# 2. Read in the book text:
filename = "sherlock_holmes_1.txt"
file = open(filename, "r", encoding="utf-8")
text = file.read()
# 3. Replace newlines with spaces:
text = text.replace("\n", " ")
# 4. Initialize an NLTK tokenizer. This uses the Punkt model we downloaded previously:
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
# 5. Divide the text into sentences:
sentences = tokenizer.tokenize(text)

# In [2]: sentences
# Out[2]:
# ['To Sherlock Holmes she is always _the_ woman.',
#  'I have seldom heard him mention her under any other name.',
#  'In his eyes she eclipses and predominates the whole of her sex.',
#  'It was not that he felt any emotion akin to love for Irene Adler.',
#  'All emotions, and that one particularly, were abhorrent to his cold, precise but admirably balanced mind.',
#  'He was, I take it, the most perfect reasoning and observing machine that the world has seen, but as a lover he would have placed himself in a false position.',
#  'He never spoke of the softer passions, save with a gibe and a sneer.',
#  'They were admirable things for the observer—excellent for drawing the veil from men’s motives and actions.',
#  'But for the trained reasoner to admit such intrusions into his own delicate and finely adjusted temperament was to introduce a distracting factor which might throw a doubt upon all his mental results.',
#  'Grit in a sensitive instrument, or a crack in one of his own high-power lenses, would not be more disturbing than a strong emotion in a nature such as his.',
#  'And yet there was but one woman to him, and that woman was the late Irene Adler, of dubious and questionable memory.']

sentences_nltk = sentences

# NLTK Sentence Segmenter
############################################


############################################
# create new display fun

from pprint import pprint


def display(*args, **kwargs):
    if len(args):
        pprint(*args, **kwargs)
    else:
        print(*args, **kwargs)

#
#############################################


############################################
# Download larger language models

import spacy
spacy.cli.download('en_core_web_sm')
spacy.cli.download('en_core_web_md')
spacy.cli.download('en_core_web_lg')

# Download larger language models
############################################


############################################
# SpaCy Small Sentence Segmenter

# 1. Import the spacy package:
import spacy
# 2. Read in the book text:
filename = 'sherlock_holmes_1.txt'  # ”->''
file = open(filename, "r", encoding="utf-8")
text = file.read()
# 3. Replace newlines with spaces:
text = text.replace("\n", " ")
# 4. Initialize the spaCy engine:
nlp = spacy.load("en_core_web_sm")
# 5. Divide the text into sentences:
doc = nlp(text)
sentences = [sentence.text for sentence in doc.sents]

# In [2]: sentences
# Out[2]:
# ['To Sherlock Holmes she is always _the_ woman.',
#  'I have seldom heard him mention her under any other name.',
#  'In his eyes she eclipses and predominates the whole of her sex.',
#  'It was not that he felt any emotion akin to love for Irene Adler.',
#  'All emotions, and that one particularly, were abhorrent to his cold, precise but admirably balanced mind.',
#  'He was, I take it, the most perfect reasoning and observing machine that the world has seen, but as a lover he would have placed himself in a false position.',
#  'He never spoke of the softer passions, save with a gibe and a sneer.',
#  'They were admirable things for the observer—excellent for drawing the veil from men’s motives and actions.',
#  'But for the trained reasoner to admit such intrusions into his own delicate and finely adjusted temperament was to introduce a distracting factor which might throw a doubt upon all his mental results.',
#  'Grit in a sensitive instrument, or a crack in one of his own high-power lenses, would not be more disturbing than a strong emotion in a nature such as his.',
#  'And yet there was but one woman to him, and',
#  'that woman was the late Irene Adler, of dubious and questionable memory.']

sentences_spacy = sentences

# SpaCy Small Sentence Segmenter
############################################

############################################
# SpaCy Small vs NLTK Sentence Segmenter

all(ss == sn for ss, sn in zip(sentences_spacy, sentences_nltk))
# Out[7]: False

[ss == sn for ss, sn in zip(sentences_spacy, sentences_nltk)]
# Out[8]: [True, True, True, True, True, True, True, True, True, True, False]

display()
display('## NLTK last 2 sentences:')
display(sentences_nltk[-2:])
# ## NLTK last 2 sentences:
# ['Grit in a sensitive instrument, or a crack in one of his own high-power lenses, would not be more disturbing than a strong emotion in a nature such as his.', 'And yet there was but one woman to him, and that woman was the late Irene Adler, of dubious and questionable memory.']


display()
display('## SpaCy last 2 sentences:')
display(sentences_spacy[-2:])
# ## SpaCy last 2 sentences:
# ['And yet there was but one woman to him, and',
#  'that woman was the late Irene Adler, of dubious and questionable memory.']


# SpaCy Small vs NLTK Sentence Segmenter
############################################


############################################
# SpaCy Medium Sentence Segmenter

# 4. Initialize the spaCy engine:
nlpmd = spacy.load("en_core_web_md")
# 5. Divide the text into sentences:
docmd = nlpmd(text)
del nlpmd
sentences = [sentence.text for sentence in docmd.sents]


sentences_spacy_md = sentences

# Medium is no better than Small for this sentence...

# In [8]: sentences_spacy_md[-2:]
# Out[8]:
# ['And yet there was but one woman to him, and',
#  'that woman was the late Irene Adler, of dubious and questionable memory.']

# SpaCy Medium Sentence Segmenter
############################################


############################################
# SpaCy Large Sentence Segmenter
#  fixes poor sentence segementation in small and medium

# filename = 'sherlock_holmes_1.txt'  # ”->''
# file = open(filename, "r", encoding="utf-8")
# text = file.read()
# # 3. Replace newlines with spaces:
# text = text.replace("\n", " ")

# 4. Initialize the spaCy engine:
nlplg = spacy.load("en_core_web_lg")
# 5. Divide the text into sentences:
doclg = nlplg(text)
del nlplg
sentences_spacy_lg = [sentence.text for sentence in doclg.sents]

display()
display('## Large SpaCy Language Model doesnt work well:')
display(sentences_spacy_lg[-1])

# filename = 'sherlock_holmes_1.txt'  # ”->''
file = open(filename, "r", encoding="utf-8")
text = file.read()
# 3. Replace newlines with spaces:
text = text.replace("\n", " ")

# 4. Initialize the spaCy engine:
nlplg = spacy.load("en_core_web_lg")
# 5. Divide the text into sentences:
doclg = nlplg(text)
del nlplg
sentences_spacy_lg = [sentence.text for sentence in doclg.sents]
print()
display('## Reloading Large SpaCy Language Model works well:')
display(sentences_spacy_lg[-2:])

# In [8]: sentences_spacy_md[-2:]
# Out[8]:
# ['And yet there was but one woman to him, and',
#  'that woman was the late Irene Adler, of dubious and questionable memory.']

# SpaCy Large Sentence Segmenter
############################################


############################################
#  Large Spacy Model "fixes" the last sentence segmentation

display()
display('## NLTK (regex) Language Model works well:')
display(sentences_nltk[-1])
# Out[6]:
# ['And yet there was but one woman to him, and that woman was the late Irene Adler, of dubious and questionable memory.']

display()
display('## Small SpaCy Language Model makes mistakes on long sentences:')
display(sentences_spacy[-1])
# Out[9]:
# 'And yet there was but one woman to him, and that woman was the late Irene Adler, of dubious and questionable memory.'

display()
display('## Medium SpaCy Language Model makes same mistake:')
display(sentences_spacy_md[-1])
# Out[6]:
# ['And yet there was but one woman to him, and',
#  'that woman was the late Irene Adler, of dubious and questionable memory.']

display()
display('## Large SpaCy Language Model works well:')
display(sentences_spacy_lg[-1])
# Out[6]:
# ['And yet there was but one woman to him, and that woman was the late Irene Adler, of dubious and questionable memory.']

# '## NLTK (regex) Language Model works well:'
# ('And yet there was but one woman to him, and that woman was the late Irene '
#  'Adler, of dubious and questionable memory.')

# '## Small SpaCy Language Model makes mistakes on long sentences:'
# 'that woman was the late Irene Adler, of dubious and questionable memory.'

# '## Medium SpaCy Language Model makes same mistake:'
# 'that woman was the late Irene Adler, of dubious and questionable memory.'

# '## Large SpaCy Language Model works well:'
# ('And yet there was but one woman to him, and that woman was the late Irene '
#  'Adler, of dubious and questionable memory.')

#  Large Spacy Model "fixes" the last sentence segmentation
############################################


################################################
# Time could be explicitly defined in text

import time
nlplg = spacy.load("en_core_web_lg")
# 5. Divide the text into sentences:
start = time.time()
sentences = [s for s in nlplg(text).sents]
stop = time.time()
del nlplg
display("SpaCy Large Time: %s s" % (stop - start))


start = time.time()
sentences = [s for s in tokenizer.tokenize(text)]
stop = time.time()
display("NLTK Large Time: %s s" % (stop - start))

# 'SpaCy Large Time: 0.021464109420776367 s'
# 'NLTK Large Time: 0.0003814697265625 s'

#
################################################
