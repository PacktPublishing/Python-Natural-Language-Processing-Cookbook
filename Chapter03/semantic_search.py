from whoosh.fields import Schema, TEXT, KEYWORD, ID, STORED, DATETIME
from whoosh.index import create_in
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import MultifieldParser
import whoosh.index
import csv
from Chapter03.word_embeddings import w2vec_model_path
from Chapter03.word_embeddings import load_model


imdb_dataset_path = "Chapter03/IMDB-Movie-Data.csv"
search_engine_index_path = "Chapter03/whoosh_index"

class IMDBSearchEngine:

    def __init__(self, index_path, imdb_path="", load_existing=False):
        self.schema = self.create_schema()
        if (not load_existing and imdb_path):
            self.data = self.read_in_csv(imdb_path)
            self.create_and_populate_index(index_path)
        elif (load_existing):
            self.load_existing_index(index_path)
        else:
            raise Exception("You need to provide the index path, and either load_existing=True or the path to the data and load_existing=False")


    def read_in_csv(self, csv_file):
        with open(csv_file, 'r', encoding='utf-8') as fp:
            reader = csv.reader(fp, delimiter=',', quotechar='"')
            data_read = [row for row in reader]
        return data_read

    def create_and_populate_index(self, index_path):
        self.index = create_in(index_path, self.schema)
        self.writer = self.index.writer()
        self.populate_index()

    def load_existing_index(self, index_path):
        self.index = whoosh.index.open_dir(index_path)

    def create_schema(self):
        schema = Schema(movie_id=ID(stored=True),
                title=TEXT(analyzer=StemmingAnalyzer()),
                description=TEXT(analyzer=StemmingAnalyzer()),
                genre=KEYWORD,
                director=TEXT,
                actors=TEXT,
                year=DATETIME)
        return schema

    def populate_index(self):
        for row in self.data[1:]:
            movie_id = row[0]
            title = row[1]
            genre = row[2]
            description = row[3]
            director = row[4]
            actors = row[5]
            year = row[6]
            self.writer.add_document(movie_id=movie_id, title=title, description=description, genre=genre, director=director, actors=actors, year=year)
        self.writer.commit()

    def query_engine(self, keywords):
        with self.index.searcher() as searcher:
            query = MultifieldParser(["title", "description"], self.index.schema).parse(keywords)
            results = searcher.search(query)
            print(results)
            print(results[0])
            return results


def get_similar_words(model, search_term):
    similarity_list = model.most_similar(search_term, topn=3)
    similar_words = [sim_tuple[0] for sim_tuple in similarity_list]
    return similar_words

def main():
    search_engine = IMDBSearchEngine(search_engine_index_path, imdb_dataset_path, load_existing=False)
    #search_engine = IMDBSearchEngine(search_engine_index_path, load_existing=True)
    model = load_model(w2vec_model_path)
    search_term = "gigantic"
    other_words = get_similar_words(model, search_term)
    results = search_engine.query_engine(" OR ".join([search_term] + other_words))


if (__name__ == "__main__"):
    main()
