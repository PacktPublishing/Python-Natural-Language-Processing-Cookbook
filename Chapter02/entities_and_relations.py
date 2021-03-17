import spacy
import textacy
from Chapter02.split_into_clauses import find_root_of_sentence

nlp = spacy.load('en_core_web_sm')
sentences = ["All living things are made of cells.", "Cells have organelles."]

verb_patterns = [[{"POS":"AUX"}, {"POS":"VERB"}, {"POS":"ADP"}], [{"POS":"AUX"}]]

def contains_root(verb_phrase, root):
    vp_start = verb_phrase.start
    vp_end = verb_phrase.end
    if (root.i >= vp_start and root.i <= vp_end):
        return True
    else:
        return False

def get_verb_phrases(doc):
    root = find_root_of_sentence(doc)
    verb_phrases = textacy.extract.matches(doc, verb_patterns)
    new_vps = []
    for verb_phrase in verb_phrases:
        if (contains_root(verb_phrase, root)):
            new_vps.append(verb_phrase)
    return new_vps

def longer_verb_phrase(verb_phrases):
    longest_length = 0
    longest_verb_phrase = None
    for verb_phrase in verb_phrases:
        if len(verb_phrase) > longest_length:
            longest_verb_phrase = verb_phrase
    return longest_verb_phrase

def find_noun_phrase(verb_phrase, noun_phrases, side):
    for noun_phrase in noun_phrases:
        if (side == "left" and noun_phrase.start < verb_phrase.start):
            return noun_phrase
        elif (side == "right" and noun_phrase.start > verb_phrase.start):
            return noun_phrase

def find_triplet(sentence):
    doc = nlp(sentence)
    verb_phrases = list(get_verb_phrases(doc))
    noun_phrases = doc.noun_chunks
    verb_phrase = None
    if (len(verb_phrases) > 1):
        verb_phrase = longer_verb_phrase(list(verb_phrases))
    else:
        verb_phrase = verb_phrases[0]
    left_noun_phrase = find_noun_phrase(verb_phrase, noun_phrases, "left")
    right_noun_phrase = find_noun_phrase(verb_phrase, noun_phrases, "right")
    return (left_noun_phrase, verb_phrase, right_noun_phrase)

def main():
    for sentence in sentences:
        (left_np, vp, right_np) = find_triplet(sentence)
        print(left_np, "\t", vp, "\t", right_np)

if (__name__ == "__main__"):
    main()

