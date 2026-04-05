"""
https://www.geeksforgeeks.org/python/count-values-in-pandas-dataframe/
"""
import pandas as pd
from langdetect import detect


def number_of_non_english_articles(csv_file):
    dataframe = pd.read_csv(csv_file, low_memory=False)
    print(f"Before: {len(dataframe)} articles")
    df = dataframe[dataframe['content'].apply(lambda article_content: detect(article_content) == 'en')]  # AI
    print(f"After: {len(df)} articles")
    print(len(dataframe))


def size_of_dataframe(csv_file):
    dataframe = pd.read_csv(csv_file, low_memory=False)
    print(f"Length of dataframe {len(dataframe)}")


def class_balance(csv_file):
    dataframe = pd.read_csv(csv_file, low_memory=False)
    classifier_labels = ['fake', 'satire', 'bias', 'conspiracy', 'junksci', 'hate', 'clickbait', 'unreliable',
                         'political',
                         'reliable']
    number_of_all_classes = dataframe['classification'].value_counts()
    for class_type in classifier_labels:
        number_of_given_class = number_of_all_classes.get(class_type, 0)
        print(f"{class_type}: {number_of_given_class}")


def number_of_missing_data(csv_file):
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
