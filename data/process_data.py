import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    """
    Load data from CSV files and merge them into a single DataFrame.

    Parameters:
    messages_filepath (str): Filepath to the CSV file containing messages data.
    categories_filepath (str): Filepath to the CSV file containing categories data.

    Returns:
    pandas.DataFrame: A DataFrame containing the merged data from messages and categories.
    
    """

    # loading datasets
    messages = pd.read_csv(messages_filepath, sep=',')
    categories = pd.read_csv(categories_filepath, sep=',')

    #merging datasets
    df = pd.merge(left=messages, right=categories, on='id', how='inner')

    return df


def clean_data(df):
    
    """
    Clean and prepare the DataFrame by splitting the 'categories' column,
    renaming columns, converting values to numeric, dropping duplicates, and 
    returning the cleaned DataFrame.

    Parameters:
    df (pandas.DataFrame): DataFrame containing messages and catagory data

    Returns:
    pandas.DataFrame: Cleaned DataFrame.
    """

    # creating a dataset from the category column
    categories = df.categories.str.split(';', expand=True)

    # renaming columns based on the first row
    row = categories.iloc[0]
    category_colnames = pd.Series(row).apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] =  categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df.drop(columns='categories', inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # dropping those observations valued as “2”
    df.drop(df[df['related'] == 2].index, inplace=True)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df   


def save_data(df, database_filename):

    """
    Save the DataFrame to a SQLite database.

    Parameters:
    df (pandas.DataFrame): Cleaned dataframe containing messages and category data
    database_filename (str): The filename for the SQLite database.
    """

    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_response', engine, if_exists = 'replace',index=False)


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

# python process_data.py messages.csv categories.csv DisasterResponse.db