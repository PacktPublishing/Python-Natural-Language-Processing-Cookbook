import spacy

sentences = ['I have seldom heard him mention her under any other name.', 
             'In his eyes she eclipses and predominates the whole of her sex.']
nlp = spacy.load('en_core_web_sm')

def get_dependency_parse(sentence):
    doc = nlp(sentence)
    for token in doc:
        print(token.text, "\t", token.dep_, "\t", spacy.explain(token.dep_))
        #print(token.text)
        #ancestors = [t.text for t in token.ancestors]
        #print(ancestors)
        #children = [t.text for t in token.children]
        #print(children)
        #subtree = [t.text for t in token.subtree]
        #print(subtree)

def main():
    get_dependency_parse(sentences[0])

if (__name__ == "__main__"):
    main()
