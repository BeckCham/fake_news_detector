"""
Filename: classifiers.py
Author: Beck Chamberlain
Version: 0.06
Description: This script trains, evaluates, and saves classification models
References:
    https://scikit-learn.org/stable/modules/naive_bayes.html
    https://scikit-learn.org/stable/modules/cross_validation.html
"""
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from src.models import model_comparison
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier


def train_model(model_to_train, sample_number):
    """
    IGNOERE AAA
    :param model_to_train:
    :param sample_number:
    :return:
    """
    # Loads the TF-IDF features/labels
    features = np.load(f'data/embedded/sample_{sample_number}/features_tfidf.npy')  # x
    labels = np.load(f'data/embedded/sample_{sample_number}/labels_tfidf.npy')  # y

    if model_to_train == "naive_bayes":
        print("Training Naive Bayes")
        model = MultinomialNB(
            alpha=1.0,  # Smoothing parameter
            class_prior=None  # Set class weights
        )
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
    elif model_to_train == "knn":
        print("Training KNN")
        model = KNeighborsClassifier(
            n_neighbors=5,  # Number of neighbours to consider
            weights='uniform',  # How neighbours should be weighted
            algorithm='brute',  # Type of algorithm
            leaf_size=30,  # Precision of ball-tree/kd_tree algoirthms
            metric='cosine',  # Distance formula
            n_jobs=-1  # How many CPU cores to use
        )
    elif model_to_train == "svm":
        print("Training SVM")
        model = LinearSVC(  # LinearSVC much faster than SVC on high dimensional TF-IDF data
            C=10.0,  # How simple/complex the model should be
            max_iter=1000,  # Maximum iterations to look for best solution
            random_state=3,  # Random seed
            class_weight=None  # For handling imbalanced classes
        )
    else:
        return

    model.fit(features, labels)

    # Validates and prints the model
    results = model_comparison.cross_validation_one_model(model, features, labels)

    # Prints results of cross validation
    print(f"Accuracy: {results[0]:.4f}")
    print(f"F1: {results[1]:.4f}")
    print(f"Precision: {results[2]:.4f}")
    print(f"Recall: {results[3]:.4f}")
    print(f"kappa: {results[4]:.4f}")
    print("\nConfusion Matrix:")
    model_comparison.print_cm_simple(results[5])

    # Save the trained model

    with open(f'src/models/sample_{sample_number}/{model_to_train}_model.pkl', 'wb') as file:
        pickle.dump(model, file)