"""
Filename: feature_selection.py
Author: Beck Chamberlain
Version: 0.05
Description: This script gathers certain data from a given news site and compacts it into a singular string that
             can be embedded
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.training.model_comparison import cross_validation_one_model, wilcoxen_feature_comparison, \
    wilcoxon_between_models
from itertools import combinations


def tf_idf_print_accuracies_with_change(tf_idf, labels):
    dt_model = DecisionTreeClassifier(
        # max_features=0.5,  # How many features to consider per split
    )
    rf_model = RandomForestClassifier(
        # n_estimators=200,  # Number of trees
        # min_samples_split=8,  # How many samples needed to split a node
    )
    svm_model = LinearSVC(
        # C=1,  # How simple/complex the model should be
        # random_state=3,  # Random seed
        # tol=0.1,
        # loss='squared_hinge',
        # penalty='l1',
    )
    nb_model = MultinomialNB(

    )
    knn_model = KNeighborsClassifier()
    models = [dt_model, rf_model, svm_model, nb_model, knn_model]
    model_names = ["dt_model", "rf_model", "svm_model", "nb_model", "knn_model"]
    for k in range(1000, 6001, +1000):
        print(f"K is {k}")
        # Select the best k features
        selector = SelectKBest(chi2, k=k)
        best_features = selector.fit_transform(tf_idf, labels)

        # Gets the results for that variation
        for index in range(0, 5):
            print(f"Model: {model_names[index]}")
            results_for_k_variation = cross_validation_one_model(models[index], best_features, labels)
            print(f"F1 results: {results_for_k_variation[1].mean():.4f}")


def tf_idf_feature_selection_with_varying_features(tf_idf, labels, model_type):
    """
    Finds the ideal amount and selection of tf_idf features using chi-square
    :return:
    """
    if model_type == "naive_bayes":
        model = MultinomialNB(

        )
    elif model_type == "decision_tree":
        model = DecisionTreeClassifier(

        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(

        )
    elif model_type == "knn":
        model = KNeighborsClassifier(

        )
    elif model_type == "svm":
        model = LinearSVC(

        )
    else:
        return

    baseline_results = cross_validation_one_model(model, tf_idf, labels)
    baseline_accuracy = baseline_results[0].mean()
    baseline_f1 = baseline_results[1].mean()
    print(f"baseline accuracy: {baseline_accuracy:.4}")
    print(f"baseline F1: {baseline_f1:.4}")

    best_accuracy_result = [baseline_results[0].mean(), "baseline", tf_idf]
    best_f1_result = [baseline_results[1].mean(), "baseline", tf_idf]
    for k in range(1000, 5001, +1000):
        # Select the best k features
        selector = SelectKBest(chi2, k=k)
        best_features = selector.fit_transform(tf_idf, labels)

        # Gets the results for that variation
        results_for_k_variation = cross_validation_one_model(model, best_features, labels)
        variation_accuracy_results = results_for_k_variation[0].mean()
        variation_f1_results = results_for_k_variation[1].mean()

        if variation_accuracy_results > best_accuracy_result[0]:
            best_accuracy_result[0] = variation_accuracy_results
            best_accuracy_result[1] = k
            best_accuracy_result[2] = best_features
        if variation_f1_results > best_f1_result[0]:
            best_f1_result[0] = variation_f1_results
            best_f1_result[1] = k
            best_f1_result[2] = best_features

    print(f"Best Accuracy model: {best_accuracy_result[1]}")
    wilcoxen_feature_comparison(model, "baseline", "new best accuracy", labels, tf_idf, best_accuracy_result[2])

    if best_accuracy_result[1] != best_f1_result[1]:
        print(f"Best F1 model: {best_f1_result[1]}")
        wilcoxen_feature_comparison(model, "baseline", "new best F1", labels, tf_idf, best_f1_result[2])
    else:
        print("Best accuracy model same as best F1")


def test_features_individually(tf_idf, additional_features, labels):
    """
    Tests features individually on Naive Bayes - converts to 2d cos Scikit-learn expects that shape
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

    model = MultinomialNB()

    # Make it so tf-idf is 2D and an array
    tf_idf = tf_idf.toarray()

    # Get baseline results with tf-idf using cross validation
    baseline_results = cross_validation_one_model(model, tf_idf, labels)
    print("Baseline just tf-idf")
    print(f"Accuracy: {baseline_results[0].mean():.4}")
    print(f"F1: {baseline_results[1].mean():.4}")

    # Gets which model has the best results for accuracy and f1
    current_best_accuracy = [baseline_results[0], "baseline"]
    current_best_f1 = [baseline_results[1], "baseline"]
    # Get changes with the addition of each feature
    for index, feature_name in enumerate(additional_feature_names):
        print(f"Feature: {feature_name}")
        # Takes the column and transforms it to become 2D
        feature = additional_features[:, index].reshape(-1, 1)
        # Combines tf-idf and the current feature
        combined_features = np.hstack((tf_idf, feature))

        # Gets a models performance after adding a feature using cross validation
        combined_features_results = cross_validation_one_model(model, combined_features, labels)

        # Compares the combined model with the baseline and outputs if there's a significant difference
        wilcoxon_between_models(baseline_results, combined_features_results, "baseline", feature_name)

        # Updates the best accuracy/f1 if its relevant
        if combined_features_results[0].mean() > current_best_accuracy[0].mean():
            current_best_accuracy = [combined_features_results[0].mean(), additional_feature_names[index]]
        if combined_features_results[1].mean() > current_best_f1[0].mean():
            current_best_f1 = [combined_features_results[1].mean(), additional_feature_names[index]]
        print('\n')

    print(f"Best Accuracy score: {current_best_accuracy[1]},{current_best_accuracy[0]:.4}")
    print(f"Best F1 score: {current_best_f1[1]},{current_best_f1[0]:.4}")


def find_best_combination_of_features(tf_idf, additional_features, labels):
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

    model = MultinomialNB()

    # Make it so tf-idf is 2D and an array
    tf_idf_combinable = tf_idf.toarray()

    # Get baseline results with tf-idf using cross validation
    baseline_results = cross_validation_one_model(model, tf_idf, labels)

    print("Baseline just tf-idf")
    print(f"Accuracy: {baseline_results[0].mean():.4}")
    print(f"F1: {baseline_results[1].mean():.4}")

    # Gets which model has the best results for accuracy and f1
    current_best_accuracy = [baseline_results[0], [["baseline"]]]
    current_best_f1 = [baseline_results[1], [["baseline"]]]

    """Asked AI for advice for combinations"""
    # Tries every possible combination of features
    for number_of_features in range(1, 8):
        for combination in combinations(range(7), number_of_features):
            # Get the features in the current combination
            current_combination_features = additional_features[:, combination]
            merged_current_combination = np.hstack((tf_idf_combinable, current_combination_features))
            # Get the results of that feature combination
            combination_results = cross_validation_one_model(model, merged_current_combination, labels)
            # Gets the names of all the features in the current combination
            combination_names = [additional_feature_names[feature_index] for feature_index in combination]
            # If current feature combination has better mean Accuracy/F1 then replace previous best F1/Accuracy
            if combination_results[0].mean() > current_best_accuracy[0].mean():
                current_best_accuracy = [combination_results[0], combination_names]
            if combination_results[1].mean() > current_best_f1[0].mean():
                current_best_f1 = [combination_results[1], combination_names]

    print("Best accuracy combination")
    print(round(current_best_accuracy[0].mean(), 4))
    print(current_best_accuracy[1])
    print("Best F1 combination")
    print(round(current_best_f1[0].mean(), 4))
    print(current_best_f1[1])


def check_best_combination(tf_idf, additional_features, labels):
    model = MultinomialNB()

    # Make it so tf-idf is 2D and an array
    tf_idf_combinable = tf_idf.toarray()

    # Select the best feature combination from all additional features
    best_accuracy_combination = additional_features[:, [0, 1, 2, 5]]
    best_f1_combination = additional_features[:, [0, 1, 2, 5,6]]
    # Merge additional features with TF-IDF vectors for both combinations
    merged_best_accuracy_combination = np.hstack((tf_idf_combinable, best_accuracy_combination))
    merged_best_f1_combination = np.hstack((tf_idf_combinable, best_f1_combination))
    # Used wilcoxen to compare difference between models
    wilcoxen_feature_comparison(model, "best_accuracy", "best_f1", labels,
                                merged_best_accuracy_combination, merged_best_f1_combination)

