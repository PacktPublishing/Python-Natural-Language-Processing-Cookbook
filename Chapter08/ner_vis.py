import spacy
from spacy import displacy
from pathlib import Path

nlp = spacy.load('en_core_web_sm')

def visualize(doc):
    colors = {"ORG": "green", "PERSON":"yellow"}
    options = {"colors": colors}
    displacy.serve(doc, style='ent', options=options)

def save_ent_html(doc, path):
    html = displacy.render(doc, style="ent")
    html_file= open(path, "w", encoding="utf-8")
    html_file.write(html)
    html_file.close()

def main():
    text = """iPhone 12: Apple makes jump to 5G
    Apple has confirmed its iPhone 12 handsets will be its first to work on faster 5G networks. 
    The company has also extended the range to include a new "Mini" model that has a smaller 5.4in screen. 
    The US firm bucked a wider industry downturn by increasing its handset sales over the past year. 
    But some experts say the new features give Apple its best opportunity for growth since 2014, when it revamped its line-up with the iPhone 6. 
    "5G will bring a new level of performance for downloads and uploads, higher quality video streaming, more responsive gaming, 
    real-time interactivity and so much more," said chief executive Tim Cook. 
    There has also been a cosmetic refresh this time round, with the sides of the devices getting sharper, flatter edges. 
    The higher-end iPhone 12 Pro models also get bigger screens than before and a new sensor to help with low-light photography. 
    However, for the first time none of the devices will be bundled with headphones or a charger."""
    doc = nlp(text)
    doc.user_data["title"] = "iPhone 12: Apple makes jump to 5G"
    visualize(doc)
    save_ent_html(doc, "Chapter08/ner_vis.html")

if (__name__ == "__main__"):
    main()

