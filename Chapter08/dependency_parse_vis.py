import spacy
from spacy import displacy
from pathlib import Path


nlp = spacy.load('en_core_web_sm')

def visualize(doc, is_list=False):
    options = {"add_lemma": True, 
            "compact": True, 
            "color": "green", 
            "collapse_punct": True, 
            "arrow_spacing": 20, 
            "bg": "#FFFFE6",
            "font": "Times",
            "distance": 120}
    if (is_list):
        displacy.serve(list(doc.sents), style='dep', options=options)
    else:
        displacy.serve(doc, style='dep', options=options)

def save_dependency_parse(doc, path):
    output_path = Path(path)
    svg = displacy.render(doc, style="dep", jupyter=False)
    output_path.open("w", encoding="utf-8").write(svg)


def main():
    text = "The great diversity of life evolved from less-diverse ancestral organisms."
    long_text = '''To Sherlock Holmes she is always _the_ woman. I have seldom heard him
    mention her under any other name. In his eyes she eclipses and
    predominates the whole of her sex. It was not that he felt any emotion
    akin to love for Irene Adler. All emotions, and that one particularly,
    were abhorrent to his cold, precise but admirably balanced mind. He
    was, I take it, the most perfect reasoning and observing machine that
    the world has seen, but as a lover he would have placed himself in a
    false position. He never spoke of the softer passions, save with a gibe
    and a sneer. They were admirable things for the observer—excellent for
    drawing the veil from men’s motives and actions. But for the trained
    reasoner to admit such intrusions into his own delicate and finely
    adjusted temperament was to introduce a distracting factor which might
    throw a doubt upon all his mental results. Grit in a sensitive
    instrument, or a crack in one of his own high-power lenses, would not
    be more disturbing than a strong emotion in a nature such as his. And
    yet there was but one woman to him, and that woman was the late Irene
    Adler, of dubious and questionable memory.'''
    
    doc = nlp(text)
    visualize(doc)
    save_dependency_parse(doc, "Chapter08/dependency_parse.svg")

if (__name__ == "__main__"):
    main()
