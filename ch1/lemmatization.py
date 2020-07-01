from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def lemmatize(words):
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return lemmatized_words

def main():
    words = ['leaf', 'leaves', 'booking', 'writing', 'completed', 'stemming']
    lem_words = lemmatize(words)
    lem_words.append(lemmatizer.lemmatize('loved', 'v'))
    lem_words.append(lemmatizer.lemmatize('worse', 'a'))
    print(lem_words)

if (__name__ == '__main__'):
    main()