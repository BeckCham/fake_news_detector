"""
Filename: lfr.py
Author: Beck Chamberlain
Version: 0.03
Description: Linguistic Features Representation
https://www.youtube.com/watch?v=f0pZviF6qLQ
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import pickle

from sklearn.preprocessing import MinMaxScaler
def prepare_data(csv_file):
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
    vectorizer = TfidfVectorizer(max_features=6000, stop_words='english', ngram_range=(1, 2), min_df=2)
    vectorized_combined_text = vectorizer.fit_transform(sample_dataframe['combined_text'])

    # Extracts additional textural & credibility feature columns
    additional_feature_column_names = [
        'dmarc_check',
        'uppercase_word_frequency',
        'language_diversity',
        'spelling_error_frequency',
    ]
    additional_features = sample_dataframe[additional_feature_column_names].values

    # Scale additional features with MinMaxScaler
    scaler = MinMaxScaler()
    scaled_additional_features = scaler.fit_transform(additional_features)

    #Sets the labels
    labels = sample_dataframe['classification'].values

    return vectorized_combined_text, scaled_additional_features, labels, scaler, vectorizer

def create_tfidf_dataset(vectorized_combined_text, scaled_additional_features, labels, scalar, vectorizer):
    """
    Creates dataset with TF-IDF applied to any relevant columns from sample

    :param vectorizer:
    :param vectorized_combined_text:
    :param scalar:
    :param labels:
    :param scaled_additional_features:
    :param csv_file: sample csv file
    :return: dataset with TF-IDF embedded features and textual & credibility features
    """


    # Save scaler for use at prediction time
    with open('data/embedded/scaler.pkl', 'wb') as f:
        pickle.dump(scalar, f)

    # Saves features and their associated labels
    np.save('data/embedded/labels.npy', labels)

    np.save('data/embedded/tf_idf.npy', vectorized_combined_text.toarray())

    np.save('data/embedded/additional_features.npy', scaled_additional_features)

    # Saves the vectorizer so it can be used on new webpages
    with open('data/embedded/tf_idf_vectorizer.pkl', 'wb') as file:
        pickle.dump(vectorizer, file)
