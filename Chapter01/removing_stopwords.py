import csv
import nltk
from nltk.probability import FreqDist

def read_in_csv(csv_file):
    with open(csv_file, 'r', encoding='utf-8') as fp:
        reader = csv.reader(fp, delimiter=',', quotechar='"')
        data_read = [row[0] for row in reader]
    return data_read

def read_text_file(filename):
    file = open(filename, "r", encoding="utf-8") 
    return file.read()

def preprocess_text(text):
    text = text.replace("\n", " ")
    return text

def tokenize(text):
    return nltk.tokenize.word_tokenize(text)

def create_frequency_dist(words):
    fdist = FreqDist(word.lower() for word in words)
    return fdist

def compile_stopwords_list_frequency(text, frequency=False):
    text = preprocess_text(text)
    words = tokenize(text)
    freq_dist = create_frequency_dist(words)
    words_with_frequencies = [(word, freq_dist[word]) for word in freq_dist.keys()]
    sorted_words = sorted(words_with_frequencies, key=lambda tup: tup[1])
    if (frequency):
        stopwords = [tuple[0] for tuple in sorted_words if tuple[1] > 100] # First option: use a frequency cutoff
    else:
        length_cutoff = int(0.02*len(sorted_words)) # Second option: use a percentage of the words
        stopwords = [tuple[0] for tuple in sorted_words[-length_cutoff:]] 
    return stopwords

def main():
    #stopwords = read_in_csv("stopwords.csv") - First option, read in a stopwords list
    stopwords = nltk.corpus.stopwords.words('english') # Second option, use a stopwords list provided with NLTK
    text = read_text_file("Chapter01/sherlock_holmes_1.txt")
    text = preprocess_text(text)
    words = tokenize(text)
    words = [word for word in words if word.lower() not in stopwords]
    print(words)
    text = read_text_file("Chapter01/sherlock_holmes.txt")
    words = compile_stopwords_list_frequency(text)
    print(words)

if (__name__ == '__main__'):
    main()    