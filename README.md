# Python Natural Language Processing Cookbook

<a href="https://www.packtpub.com/product/python-natural-language-processing-cookbook/9781838987312?utm_source=github&utm_medium=repository&utm_campaign=9781838987312"><img src="https://static.packt-cdn.com/products/9781838987312/cover/smaller" alt="Python Natural Language Processing Cookbook" height="256px" align="right"></a>

This is the code repository for [Python Natural Language Processing Cookbook](https://www.packtpub.com/product/python-natural-language-processing-cookbook/9781838987312?utm_source=github&utm_medium=repository&utm_campaign=9781838987312), published by Packt.

**Over 50 recipes to understand, analyze, and generate text for implementing language processing tasks**

## What is this book about?
Python is the most widely used language for natural language processing (NLP) thanks to its extensive tools and libraries for analyzing text and extracting computer-usable data. This book will take you through a range of techniques for text processing, from basics such as parsing the parts of speech to complex topics such as topic modeling, text classification, and visualization.

Starting with an overview of NLP, the book presents recipes for dividing text into sentences, stemming and lemmatization, removing stopwords, and parts of speech tagging to help you to prepare your data. You’ll then learn ways of extracting and representing grammatical information, such as dependency parsing and anaphora resolution, discover different ways of representing the semantics using bag-of-words, TF-IDF, word embeddings, and BERT, and develop skills for text classification using keywords, SVMs, LSTMs, and other techniques. As you advance, you’ll also see how to extract information from text, implement unsupervised and supervised techniques for topic modeling, and perform topic modeling of short texts, such as tweets. Additionally, the book shows you how to develop chatbots using NLTK and Rasa and visualize text data.

By the end of this NLP book, you’ll have developed the skills to use a powerful set of tools for text processing.

This book covers the following exciting features: 
* Become well-versed with basic and advanced NLP techniques in Python
* Represent grammatical information in text using spaCy, and semantic information using bag-of-words, TF-IDF, and word embeddings
* Perform text classification using different methods, including SVMs and LSTMs
* Explore different techniques for topic modeling such as K-means, LDA, NMF, and BERT
* Work with visualization techniques such as NER and word clouds for different NLP tools
* Build a basic chatbot using NLTK and Rasa
* Extract information from text using regular expression techniques and statistical and deep learning tools

If you feel this book is for you, get your [copy](https://www.amazon.com/dp/B08SRDF78Y) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" alt="https://www.packtpub.com/" border="5" /></a>

## Instructions and Navigations
All of the code is organized into folders.

The code will look like the following:
```
def add_ner_to_model(nlp):
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe("ner")
    return (nlp, ner)

```

**Following is what you need for this book:**
This book is for data scientists and professionals who want to learn how to work with text. Intermediate knowledge of Python will help you to make the most out of this book. If you are an NLP practitioner, this book will serve as a code reference when working on your projects. You will need Python 3 installed on your system. We recommend installing the Python libraries discussed in this book using ```pip```. The code snippets in the book mention the relevant command to install a given library on the Windows OS.

With the following software and hardware list you can run all code files present in the book (Chapter 1 - 8).

### Software and Hardware List

| Chapter  | Software required                                                                    | OS required                        |
| -------- | -------------------------------------------------------------------------------------| -----------------------------------|
|  1 - 8   |   Python 3.x (Jupyter Notebook / Anaconda)                               				    | Windows, Mac OS X, and Linux (Any) |

We also provide a PDF file that has color images of the screenshots/diagrams used in this book. [Click here to download it](https://static.packt-cdn.com/downloads/9781838987312_ColorImages.pdf).


### Related products <Other books you may enjoy>
* Hands-On Python Natural Language Processing [[Packt]](https://www.packtpub.com/product/hands-on-python-natural-language-processing/9781838989590) [[Amazon]](https://www.amazon.com/dp/B08BG5581Y)

* Getting Started with Google Bert [[Packt]](https://www.packtpub.com/product/getting-started-with-google-bert/9781838821593) [[Amazon]](https://www.amazon.com/dp/1838821597)

## Get to Know the Author
**Zhenya Antić** is a Natural Language Processing (NLP) professional working at Practical Linguistics Inc. She helps businesses to improve processes and increase productivity by automating text processing. Zhenya holds a PhD in linguistics from University of California Berkeley and a BS in computer science from Massachusetts Institute of Technology.


