"""
Filename: classifiers.py
Author: Beck Chamberlain
Version: 0.04
Description: This script trains, evaluates, and saves classification models
References:
    https://scikit-learn.org/stable/modules/naive_bayes.html
    https://scikit-learn.org/stable/modules/cross_validation.html
"""
import numpy as np
import pickle
from src.models import model_comparison
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier


def train_model(model_to_train, sample_number):
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