"""
https://www.geeksforgeeks.org/python/count-values-in-pandas-dataframe/
"""
import csv
import pandas as pd
from langdetect import detect


def number_of_non_english_articles(csv_file):
    """
    Prints the number of non english articles from the given csv file
    :param csv_file: The csv file to extract number of non-english articles from
    """
    # Loads the CSV file into a dataframe
    dataframe = pd.read_csv(csv_file)
    # Prints the size of the dataframe before articles are removed
    print(f"Before: {len(dataframe)} articles")
    # Removes the articles that are detected to not be in english
    english_dataframe = dataframe[dataframe['content'].apply(lambda article_content: detect(article_content) == 'en')] # AI
    # Prints the size of the dataframe after articles are removed
    print(f"After: {len(english_dataframe)} articles")
    # Prints the number of non-english articles
    print(f'The number of non-english articles: {len(dataframe - english_dataframe)}')


def size_of_dataframe(csv_file):
    """

    :param csv_file: The csv file to extract size of
    """
    # Loads the CSV file into a dataframe
    dataframe = pd.read_csv(csv_file, low_memory=False)
    print(f"Length of dataframe {len(dataframe)}")


def class_balance(csv_file):
    """

    :param csv_file: The csv file to extract class balance from
    """
    dataframe = pd.read_csv(csv_file, low_memory=False)
    classifier_labels = ['fake', 'satire', 'bias', 'conspiracy', 'junksci', 'hate', 'clickbait', 'unreliable',
                         'political','rumor','unknown','reliable']
    number_of_all_classes = dataframe['type'].value_counts()
    for class_type in classifier_labels:
        number_of_given_class = number_of_all_classes.get(class_type, 0)
        print(f"{class_type}: {number_of_given_class}")

def dataframe_analysis(csv_file):
    """

    :param csv_file: The csv file to be analysed
    """
    dataframe = pd.read_csv(csv_file, low_memory=False)
    print(dataframe.iloc[-1])
    print(dataframe.dtypes)

def number_of_missing_data(csv_file):
    """
    A
    :param csv_file:
    :return:
    """
    dataframe = pd.read_csv(csv_file, low_memory=False)

    # How many rows have NaN values
    null_counts_csv = dataframe.isnull().any(axis=1).sum()
    print(f"There are {null_counts_csv} non complete rows")

    # How many cells in the domain column have NaN values
    null_counts_domain = dataframe['domain'].isnull().sum()
    print(f"There are {null_counts_domain} null domains")

    # How many cells in the url column have NaN values
    null_counts_url = dataframe['url'].isnull().sum()
    print(f"There are {null_counts_url} null urls")

    # How many cells in the content column have NaN values
    null_counts_content = dataframe['content'].isnull().sum()
    print(f"There are {null_counts_content} null content")

    # How many cells in the title column have NaN values
    null_counts_title = dataframe['title'].isnull().sum()
    print(f"There are {null_counts_title} null title")

    # How many cells in the authors column have NaN values
    null_counts_authors = dataframe['authors'].isnull().sum()
    print(f"There are {null_counts_authors} null authors")

    # How many cells in the keywords column have NaN values
    null_counts_keywords = dataframe['keywords'].isnull().sum()
    print(f"There are {null_counts_keywords} null keywords")

    # How many cells in the meta keywords column have NaN values
    null_counts_meta_keywords = dataframe['meta_keywords'].isnull().sum()
    print(f"There are {null_counts_meta_keywords} null meta_keywords")

    # How many cells in the meta description column have NaN values
    null_counts_meta_description = dataframe['meta_description'].isnull().sum()
    print(f"There are {null_counts_meta_description} null meta_description")

    # How many cells in the tags column have NaN values
    null_counts_tags = dataframe['tags'].isnull().sum()
    print(f"There are {null_counts_tags} null tags")

    # How many cells in the summary column have NaN values
    null_counts_summary = dataframe['summary'].isnull().sum()
    print(f"There are {null_counts_summary} null summary")

    # How many cells in the source column have NaN values
    null_counts_source = dataframe['source'].isnull().sum()
    print(f"There are {null_counts_source} null source")
