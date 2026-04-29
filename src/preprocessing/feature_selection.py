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

from src.models.model_comparison import cross_validation_one_model, wilcoxen_feature_comparison, wilcoxon_between_means
from itertools import combinations

def tf_idf_feature_selection_with_varying_features(tf_idf,labels,model_type):
    """
    Finds the ideal amount and selection of tf_idf features using chi-square
    :return:
    """
    if model_type == "naive_bayes":
        model = MultinomialNB(
            alpha=1.0,  # Smoothing parameter
            class_prior=None  # Set class weights
        )
    elif model_type == "decision_tree":
        model = DecisionTreeClassifier(
            criterion='gini',  # What to use to determine split
            splitter='best',  # How to choose split
            max_depth=20,  # Limits tree depth/overfitting
            min_samples_split=10,  # How many samples needed to split node
            min_samples_leaf=5,  # Minimum samples needed for a leaf
            min_weight_fraction_leaf=0.0,  # Minimum weighted fraction in leaf
            max_features=None,  # How many features to consider per split
            random_state=3,  # Random seed
            max_leaf_nodes=None,  # Maximum number of leaves
            min_impurity_decrease=0.0,  # Split only if it decreases impurity by set amount
            class_weight=None,  # Handles imbalanced classes
            ccp_alpha=0.0  # Prunes tree
        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,  # Number of trees
            criterion='gini',  # Split quality measure
            max_depth=None,  # Max depth of trees
            min_samples_split=10,  # How many samples needed to split a node
            min_samples_leaf=5,  # Minimum samples needed for a leaf
            max_features='sqrt',  # Max features per split
            bootstrap=True,  # If sampling should be replacement
            random_state=3,  # Random seed
            class_weight=None,  # For imbalanced classes
            n_jobs=-1  # How many cpu cores to use
        )
    elif model_type == "knn":
        model = KNeighborsClassifier(
            n_neighbors=5,  # Number of neighbours to consider
            weights='uniform',  # How neighbours should be weighted
            algorithm='brute',  # Type of algorithm
            leaf_size=30,  # Precision of ball-tree/kd_tree algoirthms
            metric='cosine',  # Distance formula
            n_jobs=-1  # How many CPU cores to use
        )
    elif model_type == "svm":
        model = LinearSVC(  # LinearSVC much faster than SVC on high dimensional TF-IDF data
            C=10.0,  # How simple/complex the model should be
            max_iter=1000,  # Maximum iterations to look for best solution
            random_state=3,  # Random seed
            class_weight=None  # For handling imbalanced classes
        )
    else:
        return

    baseline_results = cross_validation_one_model(model, tf_idf, labels)
    baseline_accuracy = baseline_results[0].mean()
    baseline_f1 = baseline_results[1].mean()
    print(f"baseline accuracy: {baseline_accuracy:.4}")
    print(f"baseline F1: {baseline_f1:.4}")

    best_accuracy_result = [baseline_results[0].mean(),"baseline",tf_idf]
    best_f1_result = [baseline_results[1].mean(),"baseline",tf_idf]
    for k in range(1000,5000,+1000):
        # Select the best k features
        selector = SelectKBest(chi2,k=k)
        best_features = selector.fit_transform(tf_idf, labels)

        #Gets the results for that variation
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
    wilcoxen_feature_comparison(model,"baseline","new best accuracy",labels,tf_idf,best_accuracy_result[2])

    if best_accuracy_result[1] != best_f1_result[1]:
        print(f"Best F1 model: {best_f1_result[1]}")
        wilcoxen_feature_comparison(model, "baseline", "new best F1", labels, tf_idf, best_f1_result[2])
    else:
        print("Best accuracy model same as best F1")





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
    print(f"Accuracy: {baseline_results[0].mean():.4}")
    print(f"F1: {baseline_results[1].mean():.4}")

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
        wilcoxon_between_means(baseline_results, combined_features_results,"baseline",feature_name)

        #Updates the best accuracy/f1 if its relevant
        if combined_features_results[0].mean() > current_best_accuracy[0].mean():
            current_best_accuracy = [combined_features_results[0].mean(),additional_feature_names[index]]
        if combined_features_results[1].mean() > current_best_f1[0].mean():
            current_best_f1 = [combined_features_results[1].mean(),additional_feature_names[index]]


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
    tf_idf_combinable = tf_idf.toarray()

    # Get baseline results with tf-idf using cross validation
    baseline_results = cross_validation_one_model(model, tf_idf, labels)

    print("Baseline just tf-idf")
    print(f"Accuracy: {baseline_results[0].mean():.4}")
    print(f"F1: {baseline_results[1].mean():.4}")

    # Gets which model has the best results for accuracy and f1
    current_best_accuracy = [baseline_results[0], [["baseline"]]]
    current_best_f1 = [baseline_results[1], [["baseline"]]]

    """
    FIRST TWO LINES AI 
    """
    for number_of_features in range(1,8):
        for combination in combinations(range(7), number_of_features):
            #Get the columns in the current combination
            current_combination_features = additional_features[:,combination]
            merged_current_combination = np.hstack((tf_idf_combinable, current_combination_features))

            # Get the results of that feature combination
            combination_results = cross_validation_one_model(model, merged_current_combination, labels)

            # Gets the names of all the columns in the current combination
            combination_names = [additional_feature_names[feature_index] for feature_index in combination]

            if combination_results[0].mean() > current_best_accuracy[0].mean():
                current_best_accuracy = [combination_results[0],combination_names]
            if combination_results[1].mean() > current_best_f1[0].mean():
                current_best_f1 = [combination_results[1],combination_names]

    print("Best accuracy combination")
    print(round(current_best_accuracy[0].mean(),4))
    print(current_best_accuracy[1])
    print("Best F1 combination")
    print(round(current_best_f1[0].mean(),4))
    print(current_best_f1[1])

def check_best_combination(tf_idf,additional_features,labels):
    model = MultinomialNB(alpha=1.0)

    # Make it so tf-idf is 2D and an array
    tf_idf_combinable = tf_idf.toarray()

    best_combination_features = additional_features[:, [0,3,5,6]]
    merged_best_combination = np.hstack((tf_idf_combinable, best_combination_features))

    wilcoxen_feature_comparison(model,"baseline","feature_selection",labels,tf_idf_combinable,merged_best_combination)

