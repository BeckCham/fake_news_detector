"""
Filename: predict_from_url.py
Author: Beck Chamberlain
Version: 0.04
Description: This script predicts what type of classifier best fits the text given using a Naive Bayes model
"""
import pickle

from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import VotingClassifier

from src.ingestion.web_scraping import url_to_data

# Loads the vectorizer
with open('data/embedded/sample_1/tf_idf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Loads the scaler
with open('data/embedded/sample_1/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Loads SVM selector
with open(f'src/models/sample_1/selector_svm.pkl', 'rb') as file:
    svm_selector = pickle.load(file)

# Load voting model
with open('src/models/sample_1/voting_model_5.pkl', 'rb') as file:
    voting_model = pickle.load(file)


# Dictionary to map the numeric classifier to the appropriate string
classifier_labels = {
    0: 'fake',
    1: 'satire',
    2: 'bias',
    3: 'conspiracy',
    4: 'junksci',
    5: 'hate',
    6: 'clickbait',
    7: 'unreliable',
    8: 'political',
    9: 'reliable'
}


def predict(url):
    """
    Predicts the classification of an article given a url
    :param url: URL of the article to be classified
    :return: Dictionary containing the top three labels and the confidence
    """
    # Stores all webpage data from the url given
    webpage_data = url_to_data(url)
    # If no data can be retrieved then returns None
    if webpage_data is None:
        return None

    # Unpacks all the relevant webpage data
    dmarc_check, textual_features, text = webpage_data


    # Gets the additional features
    additional_features = [[
        dmarc_check,
        # uppercase word frequency
        textual_features[0],
        # language diversity
        textual_features[1],
        # spelling error frequency
        textual_features[2]
    ]]

    # Scale additional features
    scaled_additional_features = scaler.transform(additional_features)

    # Vectorizes the webpage text
    vectorised_text = vectorizer.transform([text])
    vectorised_text = svm_selector.transform(vectorised_text)  # reduce to 3000

    # Combines data
    combined_webpage_data = hstack((vectorised_text, csr_matrix(scaled_additional_features)))


    probabilities = voting_model.predict_proba(combined_webpage_data)[0]
    # Gets the top 3 predictions
    top_3_predictions = probabilities.argsort()[::-1][:3]

    top_3_prediction_probabilities = []
    for index in range(3):
        top_3_prediction_probabilities.append( {
            'label': classifier_labels[top_3_predictions[index]],
            'confidence': probabilities[top_3_predictions[index]],
        })

    return top_3_prediction_probabilities
