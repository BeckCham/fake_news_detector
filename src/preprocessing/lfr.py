"""
Filename: lfr.py
Author: Beck Chamberlain
Version: 0.04
Description: Linguistic Features Representation - Prepares and saves vectorized text features for machine learning
             Classification
https://www.youtube.com/watch?v=f0pZviF6qLQ
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import pickle

from sklearn.preprocessing import MinMaxScaler
def prepare_data(csv_file):
    """
    Prepares dataset for feature selection or to be converted into a form that can be fed to a machine learning model.
    Does this by applying word embedding to given datasets combined text column and scaling additional features.

    :param csv_file: Path to the CSV file that contains a cleaned fake news csv
    :return: Tuple that contains multiple fields including: vectorized text, scaled additional features, and labels in
            dataframes, as well as a scalar and a vectorizer
    """
    # Defines the classifiers with numeric labels
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
    # Loads the CSV file into a dataframe
    sample_dataframe = pd.read_csv(csv_file)

    # Map strings to numeric values
    sample_dataframe['classification'] = sample_dataframe['classification'].map(classifier_labels)  # Y for graph

    # Vectorizes the combined text with tf-idf and max vocab set to 6000, ngrams set to 2 and minimum occurrence to 2
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2), min_df=2)
    vectorized_combined_text = vectorizer.fit_transform(sample_dataframe['combined_text'])

    # Extracts additional textural & credibility features from dataframe
    additional_feature_column_names = [
        'dmarc_check',
        'exclamation_marks_per_sentences',
        'question_marks_per_sentences',
        'language_diversity'
    ]
    additional_features = sample_dataframe[additional_feature_column_names].values

    # Scale additional features with MinMaxScaler to make compatible with naive_bayes
    scaler = MinMaxScaler()
    scaled_additional_features = scaler.fit_transform(additional_features)

    #Extracts classification labels
    labels = sample_dataframe['classification'].values

    return vectorized_combined_text, scaled_additional_features, labels, scaler, vectorizer

def create_tf_idf_dataset(vectorized_combined_text, scaled_additional_features, labels, scalar, vectorizer):
    """
    Saves TF-IDF features, additional features, labels, scalar, and vectorizer individually.

    :param vectorized_combined_text: Matrix of TF-IDF vectors
    :param scaled_additional_features: Numpy array of scaled additional features
    :param labels: Numpy array of classification labels
    :param scalar: MinMaxScaler fitted for additional features
    :param vectorizer: TfidfVectorizer fitted for text embeddings
    :return: None - Saves files to disk
    """
    # Saves scaler for use when prediction needs to be done on new data
    with open('data/embedded/temp_folder/scaler.pkl', 'wb') as f:
        pickle.dump(scalar, f)
    # Saves scaled additional features
    np.save('data/embedded/temp_folder/additional_features.npy', scaled_additional_features)
    # Saves classification labels
    np.save('data/embedded/temp_folder/labels.npy', labels)
    # Saves TF-IDF embedded text after converting it to an array
    np.save('data/embedded/temp_folder/tf_idf.npy', vectorized_combined_text.toarray())
    # Saves the TF-IDF vectorizer for use transforming new text during prediction.
    with open('data/embedded/temp_folder/tf_idf_vectorizer.pkl', 'wb') as file:
        pickle.dump(vectorizer, file)
