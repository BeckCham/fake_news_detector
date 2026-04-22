"""
Filename: classifiers.py
Author: Beck Chamberlain
Version: 0.07
Description: This script trains, evaluates, and saves classification models
References:
    https://scikit-learn.org/stable/modules/naive_bayes.html
    https://scikit-learn.org/stable/modules/cross_validation.html
    https://medium.com/@pacosun/the-tuners-toolbox-grid-search-random-search-and-bayesian-optimization-unpacked-648abd7a8ff6
"""
from unittest import case

import numpy as np
import pickle

from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from src.models import model_comparison
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV

def train_model(model_to_train, sample_identifier, feature_selection):
    """
    IGNOERE AAA
    :param model_to_train:
    :param sample_number:
    :return:
    """
    # Loads the TF-IDF features/labels
    labels = np.load(f'data/embedded/sample_{sample_identifier}/labels.npy')
    vectorised_text = np.load(f'data/embedded/sample_{sample_identifier}/tf_idf.npy')
    #additional_features = np.load(f'data/embedded/sample_{sample_identifier}/additional_features.npy')

    if model_to_train == "naive_bayes":
        print("Training Naive Bayes")
        model = MultinomialNB(
            alpha=1.0,  # Smoothing parameter
            class_prior=None  # Set class weights
        )
        k = 1000
    elif model_to_train == "decision_tree":
        print("Training Decision Tree")
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
        k = 1000
    elif model_to_train == "random_forest":
        print("Training Random Forest")
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
        k = 4000
    elif model_to_train == "knn":
        print("Training KNN")
        model = KNeighborsClassifier(
            n_neighbors=5,  # Number of neighbours to consider
            weights='uniform',  # How neighbours should be weighted
            algorithm='brute',  # Type of algorithm
            metric='cosine',  # Distance formula
        )
        k = 1000
    elif model_to_train == "svm":
        print("Training SVM")
        model = LinearSVC(  # LinearSVC much faster than SVC on high dimensional TF-IDF data
            C=10.0,  # How simple/complex the model should be
            max_iter=1000,  # Maximum iterations to look for best solution
            random_state=3,  # Random seed
            class_weight=None  # For handling imbalanced classes
        )
        k = 2000
    else:
        return

    if feature_selection:
        # Selects the best k features from the vectorized text
        selector = SelectKBest(chi2, k=k)
        vectorised_text = selector.fit_transform(vectorised_text, labels)
    # Combines vectorized text and additional features
    #features = hstack([vectorised_text, csr_matrix(additional_features)]).toarray()
    ##############################DELETE
    features = vectorised_text
    model.fit(features, labels)

    # Validates and prints the model
    results = model_comparison.cross_validation_one_model(model, features, labels)

    # Prints results of cross validation
    print(f"Accuracy: {results[0].mean():.4f}")
    print(f"F1: {results[1].mean():.4f}")
    print(f"Precision: {results[2].mean():.4f}")
    print(f"Recall: {results[3].mean():.4f}")
    print(f"kappa: {results[4].mean():.4f}")
    print("\nConfusion Matrix:")
    model_comparison.print_cm_simple(results[5])

    # Save the trained model

    with open(f'src/models/sample_{sample_identifier}/{model_to_train}_model.pkl', 'wb') as file:
        pickle.dump(model, file)

def grid_search(model_to_train, sample_number):
    """
    IGNOERE AAA
    :param model_to_train:
    :param sample_number:
    :return:
    """
    # Loads the TF-IDF features/labels
    labels = np.load(f'data/embedded/sample_{sample_number}/labels.npy')
    vectorised_text = np.load(f'data/embedded/sample_{sample_number}/tf_idf.npy')
    additional_features = np.load(f'data/embedded/sample_{sample_number}/additional_features.npy')
    features = np.hstack([vectorised_text, additional_features])
    match(model_to_train):
        case 'naive_bayes':
            #Naive Bayes
            model = MultinomialNB()
            param_grid = {
                'alpha' : [0.1, 0.5, 1, 2, 5, 10]
            }
        case 'decision_tree':
            #Decision Tree
            model = DecisionTreeClassifier()
            param_grid = {
                'max_depth': [30],
                'min_samples_split': [15],
                'min_samples_leaf': [5],
                'criterion': ['gini', 'entropy'],
                'random_state': [3],
                'max_features': [None, 'sqrt', 'log2', 0.5]
            }
        case 'random_forest':
            #Random forest
            model = RandomForestClassifier()
            param_grid = {
                'criterion': ['gini', 'entropy'],
                'n_estimators': [50, 100, 150, 200],
                'max_depth': [20, 25, 30, None],
                'min_samples_split': [5, 10, 15],
                'min_samples_leaf': [2, 5, 9],
                'max_features': ['sqrt', 'log2'],
                'random_state': [3],
                'class_weight': [None, 'balanced']
            }
        case 'knn':
            #KNN
            model = KNeighborsClassifier()
            param_grid = {
                'n_neighbors': [3,5,7,9,11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'cosine']
            }
    #Perform grid search with selected model
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=10,
        scoring='f1_macro',
        verbose=1
    )
    grid_search.fit(features, labels)
    #Extract best hyperparameters
    best_model = grid_search.best_estimator_

    print(f"F1 score: {grid_search.best_score_:.4f}")
    for parameter,value in grid_search.best_params_.items():
        print(f"{parameter}: {value}")







