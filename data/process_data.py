import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories, left_on='id', right_on='id')
           
    
def clean_data(df):
    categories = df['categories'].str.split(';', expand=True)
    first_row = categories.iloc[0]                       
    category_colnames = (lambda x: x.str[0:-2])(first_row)                       
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].str[-1:]
        categories[column] = categories[column].astype('int64')                       
    
    df_new = df.drop(['categories'], axis=1)
    df_new = pd.concat([df_new, categories], axis=1)
    
    df_new['duplicated'] = df_new["message"].duplicated(keep='first')
    df_new = df_new[~df_new['duplicated']]
    df_new = df_new.drop(['duplicated'], axis=1)
                           
    return df_new
         
    
def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    engine.execute('DROP TABLE IF EXISTS Responses;')
    df.to_sql('Responses', engine, index=False)  

    
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()