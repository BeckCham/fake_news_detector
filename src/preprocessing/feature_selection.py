import numpy as np
from sklearn.naive_bayes import MultinomialNB
from src.models.model_comparison import cross_validation_one_model, wilcoxon_between_models


def test_features_individually(tf_idf,additional_features, labels):
    """
    Tests features individually on Naive Bayes
    :param tfidf:
    :param additional_features:
    :param labels:
    :return:
    """
    additional_feature_names = [
        'dmarc_check',
        'exclamation_marks_frequency',
        'question_marks_frequency',
        'uppercase_word_frequency',
        'sentence_length',
        'language_diversity',
        'spelling_error_frequency',
    ]

    model = MultinomialNB(alpha=1.0)

    # Make it so tf-idf is 2D and an array
    tf_idf = tf_idf.toarray()

    #Get baseline results with tf-idf using cross validation
    baseline_results = cross_validation_one_model(model, tf_idf, labels)
    print("Baseline just tf-idf")
    print(f"Accuracy: {baseline_results[0]:.4}")
    print(f"F1: {baseline_results[1]:.4}")

    #Gets which model has the best results for accuracy and f1
    current_best_accuracy = [baseline_results[0],"baseline"]
    current_best_f1 = [baseline_results[1],"baseline"]
    #Get changes with the addition of each feature
    for index, feature_name in enumerate(additional_feature_names):
        print(feature_name)
        # Gets the column and makes it 2D
        feature = additional_features[:,index].reshape(-1,1)
        #Combines tf-idf and the current feature
        combined_features = np.hstack((tf_idf, feature))

        #Gets the results after adding feature using cross validation
        combined_features_results = cross_validation_one_model(model, combined_features, labels)

        #Compares the combined model with the baseline and prints if theres a significant difference
        wilcoxon_between_models(baseline_results, combined_features_results,"baseline",feature_name)

        #Updates the best accuracy/f1 if its relevant
        if combined_features_results[0] > current_best_accuracy[0]:
            current_best_accuracy = [combined_features_results[0],additional_feature_names[index]]
        if combined_features_results[1] > current_best_f1[0]:
            current_best_f1 = [combined_features_results[1],additional_feature_names[index]]


    print(f"Best Accuracy score: {current_best_accuracy[1]},{current_best_accuracy[0]:.4}")
    print(f"Best F1 score: {current_best_f1[1]},{current_best_f1[0]:.4}")

def find_best_combination_of_features(tf_idf,additional_features,labels):
    """
    Finds the best combination of features using exhaustive feature search
    :return:
    """
    additional_feature_names = [
        'dmarc_check',
        'exclamation_marks_frequency',
        'question_marks_frequency',
        'uppercase_word_frequency',
        'sentence_length',
        'language_diversity',
        'spelling_error_frequency'
    ]

    model = MultinomialNB(alpha=1.0)

    # Make it so tf-idf is 2D and an array
    model_features = tf_idf.toarray()

    # Get baseline results with tf-idf using cross validation
    baseline_results = cross_validation_one_model(model, tf_idf, labels)

    print("Baseline just tf-idf")
    print(f"Accuracy: {baseline_results[0]:.4}")
    print(f"F1: {baseline_results[1]:.4}")

    # Gets which model has the best results for accuracy and f1
    current_best_accuracy = [baseline_results[0], [["baseline"]]]
    current_best_f1 = [baseline_results[1], [["baseline"]]]

    """
    FIRST TWO LINES AI 
    """
    for number_of_features in range(1,8):
        for selection in feature_selections(range(7), number_of_features):


