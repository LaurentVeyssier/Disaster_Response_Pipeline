import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load and return 2 datasets
    inputs:
        messages_filepath: path to messages dataset
        categories_filepath:  path to categories dataset
    outputs:
        messages: messages dataset
        categories: categories dataset
     """

    # load datasets
    messages = pd.read_csv(messages_filepath, index_col=0)
    categories = pd.read_csv(categories_filepath, index_col=0)

    return messages, categories


def clean_data(messages, categories):
    """combine 2 datasets, return the cleaned consolidated set
    after adjustment of categories dataset
    inputs:
        messages: raw messages dataset
        categories: raw categories dataset
    outputs:
        df: consolidated dataset
     """
    # create a dataframe of the 36 individual category columns
    categories = categories.categories.str.split(pat=';', expand=True)
    # select the first row of the categories dataframe
    first_row = categories.iloc[0, :].tolist()
    # extract a list of new column names for categories.
    category_colnames = [col_name[:-2] for col_name in first_row]
    # rename the columns of `categories`
    categories.columns = category_colnames
    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Merge the messages and  new `categories` datasets using the common id
    df = messages.join(categories, on='id')

    # drop duplicates
    df = df.drop_duplicates()

    # drop the few samples with "related"=2 to make it binary variable
    df = df.drop(df[df.related==2].index)

    # return clean consolidated dataset
    return df


def save_data(df, database_filename):
    """Save cleaned dataframe to sql database using provided path
    inputs:
        df: cleaned dataframe
        database_filename: path to database (str), ie '../data/DisasterResponse.db'
    outputs:
        None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('messages', engine, index=False, if_exists='replace')


def main():
    """pipeline to load raw data, preprocess and save cleaned dataset"""
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))

        messages, categories = load_data(
            messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages, categories)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
