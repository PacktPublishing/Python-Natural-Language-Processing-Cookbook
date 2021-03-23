import spacy
from spacy.util import minibatch, compounding
from spacy.language import Language
import warnings
import random
from pathlib import Path

DATA = [
    ("A fakir from far-away India travels to Asterix's village and asks Cacofonix to save his land from drought since his singing can cause rain.", 
        {'entities':[(39, 46, "PERSON"), (66, 75, "PERSON")]}),
    ("Cacofonix, accompanied by Asterix and Obelix, must travel to India aboard a magic carpet to save the life of the princess Orinjade, who is to be sacrificed to stop the drought.", 
        {'entities':[(0, 9, "PERSON"), (26, 33, "PERSON"), (38, 44, "PERSON"), (61, 66, "LOC"), (122, 130, "PERSON")]})
]

NEW_LABEL = "GAULISH_WARRIOR"

MODIFIED_DATA = [
    ("A fakir from far-away India travels to Asterix's village and asks Cacofonix to save his land from drought since his singing can cause rain.", 
        {'entities':[(39, 46, NEW_LABEL), (66, 75, NEW_LABEL)]}),
    ("Cacofonix, accompanied by Asterix and Obelix, must travel to India aboard a magic carpet to save the life of the princess Orinjade, who is to be sacrificed to stop the drought.", 
        {'entities':[(0, 9, NEW_LABEL), (26, 33, NEW_LABEL), (38, 44, NEW_LABEL), (61, 66, "LOC"), (122, 130, "PERSON")]})
]

N_ITER=100
OUTPUT_DIR = "Chapter05/model_output"

def load_model(input_dir):
    nlp = spacy.load(input_dir)
    return nlp

def save_model(nlp, output_dir):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()
    nlp.to_disk(output_dir)

def create_model(model):
    if (model is not None):
        nlp = spacy.load(model)
    else:
        nlp = spacy.blank("en")
    return nlp

def add_ner_to_model(nlp):
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe("ner")
    return (nlp, ner)

def add_labels(ner, data):
    for sentence, annotations in data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    return ner

def train_model(model=None):
    nlp = create_model(model)
    (nlp, ner) = add_ner_to_model(nlp)
    ner = add_labels(ner, DATA)
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        warnings.filterwarnings("once", category=UserWarning, module='spacy')
        if model is None:
            nlp.begin_training()
        for itn in range(N_ITER):
            random.shuffle(DATA)
            losses = {}
            batches = minibatch(DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,
                    annotations,
                    drop=0.5,
                    losses=losses,
                )
            print("Losses", losses)
    return nlp

def train_model_new_entity_type(model=None):
    random.seed(0)
    nlp = create_model(model)
    (nlp, ner) = add_ner_to_model(nlp)
    ner = add_labels(ner,  MODIFIED_DATA)
    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
    move_names = list(ner.move_names)
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        warnings.filterwarnings("once", category=UserWarning, module='spacy')
        sizes = compounding(1.0, 4.0, 1.001)
        for itn in range(N_ITER):
            random.shuffle(MODIFIED_DATA)
            batches = minibatch(MODIFIED_DATA, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
            print("Losses", losses)
    return nlp

def test_model(nlp, data):
    for text, annotations in data:
        doc = nlp(text)
        for ent in doc.ents:
            print(ent.text, ent.start_char, ent.end_char, ent.label_)

def without_training(data=DATA):
    nlp = spacy.load("en_core_web_sm")
    test_model(nlp, data)

def main():
    #without_training()
    model = "en_core_web_sm"
    #nlp = train_model(model)
    #nlp = train_model()
    nlp = train_model_new_entity_type(model)
    test_model(nlp, DATA)
    #save_model(nlp, OUTPUT_DIR)

def load_and_test(model_dir, data=DATA):
    nlp = load_model(model_dir)
    test_model(nlp, data)

if (__name__ == "__main__"):
    #main()
    load_and_test(OUTPUT_DIR)
