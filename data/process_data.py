import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # load two datasets and merge together as df
    # arguments: file path of message dataset; 
    #           file path of categories dataset;
    # return: merged dataset
    
    #load messages ans cetegories datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #merge datasets
    df = messages.merge(categories.drop_duplicates(subset=['id']), how='left', on='id')
    
    return df
    

def clean_data(df):
    # clean the merged dataset 
    # arguments: original dateset
    #          
    # return: cleaneded dataset
    
    # split values in categories into 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # extract a list of new column names for categories from first row and apply new name to individual columns
    category_colnames = categories.iloc[1].apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    # convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
 
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column].astype(str))
    
    # drop original categories column and concatenate with new columns 
    cleaned_df = df.drop('categories', axis=1)
    cleaned_df = pd.concat([cleaned_df, categories], axis=1)
    
    # drop duplicates by 'id'
    cleaned_df.drop_duplicates(subset='id', inplace=True)
    
    return cleaned_df


def save_data(df, database_filename):
    # Save the clean dataset into an sqlite database
    # arguments: df: cleaned dataset
    #           database_filename: database file name
    # return: none
    
    # create engine for database
    engine = create_engine('sqlite:///' + database_filename)
    
    #save cleaned dataset to SQL file named 'DisasterResponseTable'
    df.to_sql('DisasterResponseTable', engine, index=False)


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