"""
Filename: predict_from_url.py
Author: Beck Chamberlain
Version: 0.03
Description: This script predicts what type of classifier best fits the text given using a Naive Bayes model
"""
import pickle

from scipy.sparse import hstack, csr_matrix

from src.ingestion.web_scraping import url_to_data

# Loads the vectorizer
with open('data/embedded/sample_1/tf_idf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)
    """
#Loads the scaler
with open('data/embedded/sample_1/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
    """
# Loads naive bayes model
with open('src/models/no_added/naive_bayes_model.pkl', 'rb') as file:
    nb_model = pickle.load(file)
# Loads decision tree model
with open('src/models/no_added/naive_bayes_model.pkl', 'rb') as file:
    dt_model = pickle.load(file)

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


def predict(url, model_to_use):
    # Stores all webpage data from the url given
    webpage_data = url_to_data(url)
    # If no data can be retrieved then returns None
    if webpage_data is None:
        return None

    # Unpacks all the relevant webpage data
    dmarc_check, textual_features, text = webpage_data

    # Vectorizes the webpage text
    vectorised_text = vectorizer.transform([text])
    """
    # Gets the additional features
    additional_features = [[
        dmarc_check,
        # exclamation marks frequency
        textual_features[0],
        #question marks frequency
        textual_features[1],
        # uppercase word frequency
        textual_features[2],
        # average sentence length
        textual_features[3],
        # language diversity
        textual_features[4],
        # spelling error frequency
        textual_features[5]
    ]]
    
    # Scale additional features
    scaled_additional_features = scaler.transform(additional_features)

    #Combines data
    combined_webpage_data = hstack((vectorised_text, csr_matrix(scaled_additional_features)))
    """
    # Gets the most likely prediction
    if model_to_use == 'naive_bayes':
        main_prediction = nb_model.predict(vectorised_text)[0]
        #main_prediction = nb_model.predict(combined_webpage_data)[0]
        # Gets all the probabilities by percentage
        #probabilities = nb_model.predict_proba(combined_webpage_data)[0
        probabilities = nb_model.predict_proba(vectorised_text)[0]
    else:
        print("aa")
        #main_prediction = dt_model.predict(combined_webpage_data)[0]
        # Gets all the probabilities by percentage
        #probabilities = dt_model.predict_proba(combined_webpage_data)[0]

    """
    Get top 3 probabilities 
    """
    # Gets confidence in main prediction
    main_prediction_confidence = probabilities[main_prediction] * 100
    # Gets the string label of the main classifier
    main_label = classifier_labels[main_prediction]

    return {
        'label': main_label,
        'confidence': main_prediction_confidence,
    }
