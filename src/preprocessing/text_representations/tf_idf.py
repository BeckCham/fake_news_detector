"""
Filename: predict_from_url.py
Author: Beck Chamberlain
Version: 0.02
Description: Embeds a cleaned sample into tf_idf
https://www.youtube.com/watch?v=f0pZviF6qLQ
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import pickle


def create_tfidf_dataset(csv_file):
    """
    Creates dataset with TF-IDF applied to any relevant columns from sample

    :param csv_file: sample csv file
    :return: dataset with TF-IDF embedded features and textual & credibility features
    """
    # Defines the classifiers numerically
    classifier_labels = {
        'fake': 0,
        'satire': 1,
        'bias': 2,
        'conspiracy': 3,
        'junksci': 4,
        'hate': 5,
        'clickbait': 6,
        'unreliable': 7,
        'political': 8,
        'reliable': 9,
    }
    # Loads the csv_file to be embedded into a dataframe
    sample_dataframe = pd.read_csv(csv_file)

    # Map strings to numeric values
    sample_dataframe['classification'] = sample_dataframe['classification'].map(classifier_labels)  # Y for graph

    # Vectorizes the combined text with tf-idf and max vocab set to 5000, ngrams set to 2 and minimum occurrence to 2
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2), min_df=2)
    vectorized_combined_text = vectorizer.fit_transform(sample_dataframe['combined_text'])

    # Extracts additional textural & credibility feature columns
    additional_feature_column_names = [
        'dmarc_check',
        'exclamation_marks_frequency',
        'question_marks_frequency',
        'uppercase_word_frequency',
        'sentence_length',
        'language_diversity',
        'spelling_error_frequency'
    ]
    additional_features = sample_dataframe[additional_feature_column_names].values

    # Combine vectorised text and additional features
    from scipy.sparse import hstack, csr_matrix
    combined_features =hstack([vectorized_combined_text, csr_matrix(additional_features)])

    # Saves features and their associated labels
    np.save('data/embedded/labels_tfidf.npy', sample_dataframe['classification'].values)
    np.save('data/embedded/features_tfidf.npy', vectorized_combined_text.toarray())

    # Saves the vectorizer so it can be used on new webpages
    with open('data/embedded/tfidf_vectorizer.pkl', 'wb') as file:
        pickle.dump(vectorizer, file)
