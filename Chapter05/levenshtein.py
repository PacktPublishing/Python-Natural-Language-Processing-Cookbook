import pandas as pd
import Levenshtein
from Chapter05.regex import get_emails

data_file = "Chapter05/DataScientist.csv"

def find_levenshtein(input_string, df):
    df['distance_to_' + input_string] = df['emails'].apply(lambda x: Levenshtein.distance(input_string, x))
    return df

def find_jaro(input_string, df):
    df['distance_to_' + input_string] = df['emails'].apply(lambda x: Levenshtein.jaro(input_string, x))
    return df

def get_closest_email_lev(df, email):
    df = find_levenshtein(email, df)
    column_name = 'distance_to_' + email
    minimum_value_email_index = df[column_name].idxmin()
    email = df.loc[minimum_value_email_index]['emails']
    return email

def get_closest_email_jaro(df, email):
    df = find_jaro(email, df)
    column_name = 'distance_to_' + email
    maximum_value_email_index = df[column_name].idxmax()
    email = df.loc[maximum_value_email_index]['emails']
    return email    

def main():
    df = pd.read_csv(data_file, encoding='utf-8')
    emails = get_emails(df)
    new_df = pd.DataFrame(emails,columns=['emails'])
    input_string = "rohitt.macdonald@prelim.com"
    email = get_closest_email_jaro(new_df, input_string)
    print(email)

if (__name__ == "__main__"):
    main()