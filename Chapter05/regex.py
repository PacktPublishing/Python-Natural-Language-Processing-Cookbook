import re
import pandas as pd

data_file = "Chapter05/DataScientist.csv"

def get_items(df, regex, column_name):
    df[column_name] = df['Job Description'].apply(lambda x: re.findall(regex, x))
    return df

def get_list_of_items(df, column_name):
    items = []
    for index, row in df.iterrows():
        if (len(row[column_name]) > 0):
            for item in list(row[column_name]):
                if (type(item) is tuple and len(item) > 1):
                    item = item[0]
                if (item not in items):
                    items.append(item)
    return items

def get_emails(df):
    original_email_regex = '[\S]+@[a-zA-Z0-9\.]+\.[a-zA-Z]+'
    email_regex = '[^\s:|()\']+@[a-zA-Z0-9\.]+\.[a-zA-Z]+'
    df['emails'] = df['Job Description'].apply(lambda x: re.findall(email_regex, x))
    emails = get_list_of_items(df, 'emails')
    return emails

def get_urls(df):
    url_regex = '(http[s]?://(www\.)?[A-Za-z0-9–_\.\-]+\.[A-Za-z]+/?[A-Za-z0-9$\–_\-\/\.]*)[\.)\"]*'
    df = get_items(df, url_regex, 'urls')
    urls = get_list_of_items(df, 'urls')
    return urls


def main():
    df = pd.read_csv(data_file, encoding='utf-8')
    emails = get_emails(df)
    print(emails)
    print(len(emails))
    urls = get_urls(df)
    print(urls)
    print(len(urls))


if (__name__ == "__main__"):
    main()

