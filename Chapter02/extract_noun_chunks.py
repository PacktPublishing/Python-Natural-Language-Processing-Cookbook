import spacy
from Chapter01.dividing_into_sentences import read_text_file
from Chapter01.dividing_into_sentences import divide_into_sentences_nltk
from Chapter01.dividing_into_sentences import preprocess_text

nlp = spacy.load('en_core_web_md')

def explore_properties(sentence):
    doc = nlp(sentence)
    other_span = "emotions"
    other_doc = nlp(other_span)
    for noun_chunk in doc.noun_chunks:
        print(noun_chunk.text)
        #print(noun_chunk.text, "\t", noun_chunk.start, "\t", noun_chunk.end)
        #print(noun_chunk.sent)
        #print(noun_chunk.root.text)
        print(noun_chunk.similarity(other_doc))
    #print(doc.similarity(other_doc))


def get_sherlock_holmes_noun_chunks():
    text = read_text_file("Chapter01/sherlock_holmes_1.txt")
    text = preprocess_text(text)
    doc = nlp(text)
    for noun_chunk in doc.noun_chunks:
        print(noun_chunk.text)

def main():
    sentence = "All emotions, and that one particularly, were abhorrent to his cold, precise but admirably balanced mind."
    explore_properties(sentence)

if (__name__ == "__main__"):
    main()

#other_span = "emotions"
#other_doc = nlp(other_span)
#for noun_chunk in doc.noun_chunks:
#    print(noun_chunk.similarity(other_doc))