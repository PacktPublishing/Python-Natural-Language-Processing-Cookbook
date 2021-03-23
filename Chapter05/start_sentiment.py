from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sentences = ["I love going to school!", "I hate going to school!"]
sid = SentimentIntensityAnalyzer()

def get_blob_sentiment(sentence):
    result = TextBlob(sentence).sentiment
    print(sentence, result.polarity)
    return result.polarity

def get_nltk_sentiment(sentence):
    ss = sid.polarity_scores(sentence)
    print(sentence, ss['compound'])
    return ss['compound']

def main():
    for sentence in sentences:
        sentiment = get_blob_sentiment(sentence)

if (__name__ == "__main__"):
    main()

