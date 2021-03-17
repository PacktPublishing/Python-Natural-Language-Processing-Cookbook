import os
import nltk
from os import path
#from PIL import Image
#import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from Chapter01.dividing_into_sentences import read_text_file
from Chapter01.removing_stopwords import compile_stopwords_list_frequency

def create_wordcloud(text, stopwords, filename, apply_mask=None):
    if (apply_mask is not None):
        wordcloud = WordCloud(background_color="white", max_words=2000, mask=apply_mask,
                              stopwords=stopwords, min_font_size=10, max_font_size=100)
        wordcloud.generate(text)
        wordcloud.to_file(filename)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.figure()
        plt.imshow(apply_mask, cmap=plt.cm.gray, interpolation='bilinear')
        plt.axis("off")
        plt.show()
    else:
        wordcloud = WordCloud(min_font_size=10, max_font_size=100, stopwords=stopwords, width=1000, height=1000, max_words=1000, background_color="white").generate(text)
        wordcloud.to_file(filename)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

def main():
    text_file = "Chapter01/sherlock_holmes.txt"
    text = read_text_file(text_file)
    #sherlock_data = Image.open("Chapter08/sherlock.png")
    #sherlock_mask = np.array(sherlock_data)
    create_wordcloud(text, compile_stopwords_list_frequency(text), "Chapter08/sherlock_wc.png")
    #create_wordcloud(text, compile_stopwords_list_frequency(text), "Chapter08/sherlock_mask.png", apply_mask=sherlock_mask)

if (__name__ == "__main__"):
    main()


