from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer('english')
#stemmer = SnowballStemmer('spanish')


def stem(word):
    stem = stemmer.stem(word)
    return stem

def main():
    words = ['leaf', 'leaves', 'booking', 'writing', 'completed', 'stemming', 'skiing', 'skies']
    #spanish_words = ['caminando', 'amigo', 'bueno']
    stemmed_words = [stem(word) for word in words]
    print(stemmed_words)

if (__name__ == '__main__'):
    main()