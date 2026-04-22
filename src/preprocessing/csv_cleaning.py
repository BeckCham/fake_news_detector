"""
Filename: csv_preprocessing.py
Author: Beck Chamberlain
Version: 0.01
Description: This script performs preprocessing techniques on a given file.
References:
https://note.nkmk.me/en/python-ast-literal-eval/
"""
import ast
import pandas as pd
import requests
from bs4 import BeautifulSoup
from langdetect import detect
from src.preprocessing import added_features
from src.preprocessing.added_features import dmarc_check, html_w3c_compliance


def is_article_not_english(content):
    """

    :param content:
    :return:
    """
    try:
        # Gets the first detected language
        return detect(content) != 'en'
    except:
        # Assume not english if error occurs
        return True


def remove_non_english_articles(dataframe):
    dataframe = dataframe[
        dataframe['content'].apply(lambda article_content: not is_article_not_english(article_content))]
    return dataframe


def cleans_meta_keywords_column(cell):
    """
    Goes through a cell in the meta keywords column and safely converts the metadata keywords into a string and empty
    keywords are removed.

    :param cell: Meta keywords cell
    :return: Cleaned meta keywords column
    """
    try:
        parsed_list = ast.literal_eval(cell)
        # Removes empty keywords
        filtered_list = [meta_keyword for meta_keyword in parsed_list if meta_keyword != '']
        # If the list is empty then its set as null
        return ' '.join(filtered_list) if filtered_list else None
    except (ValueError, SyntaxError, TypeError):  # If value is NaN
        return None


def culling_redundant_features(dataframe):
    """
    Drops all redundant columns from the given dataframe

    :param dataframe: The dataframe to be culled
    :return: A dataframe without redundant columns
    """
    # Removed summary due to its frequently null in the dataset
    dataframe = dataframe.drop(columns=['summary'])
    # Removed source because it doesn't give further information than the domain when it is present
    dataframe = dataframe.drop(columns=['source'])
    # Removed because it repeats same information as domain but with noise, if extra time is granted could be tokenised
    dataframe = dataframe.drop(columns=['url'])
    # Removed because in all 3 of my samples the keywords column was always fully null
    dataframe = dataframe.drop(columns=['keywords'])

    # Remove the following as the information is irrelevant to the predictive model
    dataframe = dataframe.drop(columns=['id'])
    dataframe = dataframe.drop(columns=['scraped_at'])
    dataframe = dataframe.drop(columns=['inserted_at'])
    dataframe = dataframe.drop(columns=['updated_at'])
    return dataframe


def combining_textual_features(dataframe):
    """
    Combines textual features into one column, adding weight to more important text features

    :param dataframe: dataframe to be combined
    :return: dataframe with combined textual features
    """
    # Goes through all the columns to be combined and ensures every cell within them have a string value
    for column in ['title', 'tags', 'meta_keywords', 'authors', 'content', 'meta_description']:
        dataframe[column] = dataframe[column].fillna('')  # Converts NaN value to ' '

    dataframe['combined_text'] = (
        # Adds the domain name twice as it can contain important words eg "patriot"
            #dataframe['domain'] + ' ' + dataframe['domain'] + ' ' +
            # Adds the title twice as its shown to be a good indicator of the validity of news
            dataframe['title'] + ' ' + dataframe['title'] + ' ' +
            # Adds the tags/meta keywords twice as it will emphasize keywords further
            dataframe['tags'] + ' ' + dataframe['tags'] + ' ' +
            dataframe['meta_keywords'] + ' ' + dataframe['meta_keywords'] + ' ' +
            # Adds the author twice, having a normal name vs "the daily sheeple" is very helpful in determining validity
            dataframe['authors'] + ' ' + dataframe['authors'] + ' ' +
            dataframe['content'] + ' ' +
            dataframe['meta_description']
    )
    # Remove the text based columns which can add no new information
    dataframe = dataframe.drop(columns=['meta_description'])
    dataframe = dataframe.drop(columns=['content'])
    dataframe = dataframe.drop(columns=['title'])
    dataframe = dataframe.drop(columns=['meta_keywords'])
    dataframe = dataframe.drop(columns=['tags'])
    # Remove the author and domain column to try to encourage the model to learn textual analysis rather than source credibility
    dataframe = dataframe.drop(columns=['authors'])
    dataframe = dataframe.drop(columns=['domain'])
    return dataframe

def add_extra_features(dataframe):
    # Adds a dmarc check
    dataframe['dmarc_check'] = dataframe['domain'].apply(added_features.dmarc_check)
    # Gets textual analysis information and adds features to the dataset
    textual_features = dataframe['content'].apply(added_features.textual_analysis)
    dataframe['uppercase_word_frequency'] = textual_features.apply(lambda x: x[0])
    dataframe['language_diversity'] = textual_features.apply(lambda x: x[1])
    dataframe['spelling_error_frequency'] = textual_features.apply(lambda x: x[2])

    return dataframe


def preprocess_csv(csv_file):
    """
    Preprocesses the given csv file using cleaning, culling, and embedding methods.
    :param csv_file: The csv file to be preprocessed
    """
    # Converts the csv file into a dataframe using low memory to ensure columns are classified properly
    dataframe = pd.read_csv(csv_file, low_memory=False)
    # Removes all non english articles
    dataframe = remove_non_english_articles(dataframe)
    # Cleans the meta keywords field
    dataframe['meta_keywords'] = dataframe['meta_keywords'].apply(cleans_meta_keywords_column)
    # Adds extra features to the dataframe
    dataframe = add_extra_features(dataframe)
    # Culls all redundant features from the dataframe
    dataframe = culling_redundant_features(dataframe)
    dataframe = combining_textual_features(dataframe)
    # Prepares the dataframe for merging
    dataframe.rename(columns={'type': 'classification'}, inplace=True)
    # Saves the cleaned dataframe as a csv
    dataframe.to_csv('data/cleaned/cleaned_sample.csv', index=False)
