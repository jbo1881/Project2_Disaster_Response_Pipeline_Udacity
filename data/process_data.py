import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories data from CSV files.

    Parameters:
        messages_filepath (str): Filepath to the messages CSV file.
        categories_filepath (str): Filepath to the categories CSV file.

    Returns:
        df (pd.DataFrame): A DataFrame containing merged data with messages and categories.
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')

    return df

def clean_data(df):
    """
    Clean and preprocess a DataFrame containing message categories data.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing message categories.

    Returns:
        df (pd.DataFrame): Cleaned and preprocessed DataFrame with categories as numeric values.
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    categories.head()

    # select the first row of the categories dataframe
    row = categories.head(1)

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.applymap(lambda cell: cell[:-2]).iloc[0, :]
    category_colnames = category_colnames.tolist()

    # rename the columns of `categories`
    categories.columns = category_colnames

    # convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df



def save_data(df, database_filename):
    """
    Save a DataFrame to a SQLite database.

    Parameters:
        df (pd.DataFrame): DataFrame to be saved to the database.
        database_filename (str): Filepath to the SQLite database.

    Returns:
        None
    """
    engine = create_engine('sqlite:///Disaster_Messages.db')
    df.to_sql('Disaster_Messages', engine, index=False)  


def main():
    """
    Main function for loading, cleaning, and saving disaster response data.

    Usage:
    python process_data.py <messages_filepath> <categories_filepath> <database_filepath>

    Parameters:
        messages_filepath (str): Filepath to the messages CSV file.
        categories_filepath (str): Filepath to the categories CSV file.
        database_filepath (str): Filepath to the SQLite database to save the cleaned data.

    Returns:
        None
    """
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