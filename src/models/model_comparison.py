"""
Filename: model_comparison.py
Author: Beck Chamberlain
Version: 0.02
Description:
"""
from sklearn.metrics import confusion_matrix, cohen_kappa_score, make_scorer
from sklearn.model_selection import cross_val_score, cross_val_predict
from scipy import stats


def cross_validation_one_model(model, features, labels):
    # Uses 10-fold cross validation
    accuracy_result = cross_val_score(model, features, labels, cv=10, scoring='accuracy')
    f1_result = cross_val_score(model, features, labels, cv=10, scoring='f1_weighted')
    precision_result = cross_val_score(model, features, labels, cv=10, scoring='precision_weighted')
    recall_result = cross_val_score(model, features, labels, cv=10, scoring='recall_weighted')
    # Calculates kappa
    kappa_scorer = make_scorer(cohen_kappa_score)
    kappa_result = cross_val_score(model, features, labels, cv=10, scoring=kappa_scorer)
    # Calculates confusion matrix
    whole_prediction = cross_val_predict(model, features, labels, cv=10)
    whole_confusion_matrix = confusion_matrix(labels, whole_prediction)
    # Return results
    return accuracy_result.mean(), f1_result.mean(), precision_result.mean(), recall_result.mean(), kappa_result.mean(), whole_confusion_matrix


def cross_validation_two_models(model_1, model_1_name, model_2, model_2_name, features, labels):
    # Gets the cross validated metrics for both models
    cross_val_result_1 = cross_validation_one_model(model_1, features, labels)
    cross_val_result_2 = cross_validation_one_model(model_2, features, labels)

    # Prints comparisons by subtracting models 2's results from model 1's
    print_comparison(cross_val_result_1, cross_val_result_2, model_1_name, model_2_name)

def cross_validation_two_samples(model_1, model_2, model_1_name, model_2_name, features_1, labels_1, features_2,
                                 labels_2):
    # Gets the cross validated metrics for both models
    cross_val_result_1 = cross_validation_one_model(model_1, features_1, labels_1)
    cross_val_result_2 = cross_validation_one_model(model_2, features_2, labels_2)

    # Prints comparisons by subtracting models 2's results from model 1's
    print_comparison(cross_val_result_1, cross_val_result_2, model_1_name, model_2_name)

"""
Does related Wilcoxon signed-rank tests between two different models
"""
def wilcoxon_between_models(model_1_results, model_2_results, model_1_name, model_2_name):
    # Prints if there's a significant difference in the two models accuracy
    print("Accuracy:")
    print(f"{model_1_name}:{model_1_results[0]:.4f}, {model_2_name}:{model_2_results[0]:.4f} ")
    stat, p_value = stats.wilcoxon(model_1_results[0], model_2_results[0])
    print_if_significant(p_value)
    # Prints if there's a significant difference in the two models F1
    print("F1:")
    print(f"{model_1_name}:{model_1_results[1]:.4f}, {model_2_name}:{model_2_results[1]:.4f} ")
    stat, p_value = stats.wilcoxon(model_1_results[1], model_2_results[1])
    print_if_significant(p_value)
    # Prints if there's a significant difference in the two models kappa
    print("Kappa:")
    print(f"{model_1_name}:{model_1_results[4]:.4f}, {model_2_name}:{model_2_results[4]:.4f} ")
    stat, p_value = stats.wilcoxon(model_1_results[4], model_2_results[4])
    print_if_significant(p_value)
"""
Does Wilcoxen signed-rank tests between two different or samples
"""
def t_test_between_samples(model_1_results, model_2_results, model_1_name, model_2_name):
    # Prints if there's a significant difference in the two models accuracy
    print("Results:")
    print(f"{model_1_name}:{model_1_results:.4f}, {model_2_name}:{model_2_results:.4f} ")
    stat, p_value = stats.wilcoxon(model_1_results, model_2_results)
    print_if_significant(p_value)

def print_if_significant(p_value):
    if p_value < 0.05:
        print("There is a significant difference between the models ")
    else:
        print("There is not a significant difference")

def print_comparison(cross_val_result_1, cross_val_result_2, model_1_name, model_2_name):
    # Prints comparisons by subtracting model 2's results from model 1's
    print(f"Comparison showing {model_1_name} - {model_2_name}: ")
    print(f"Accuracy: {cross_val_result_1[0]:.4f} - {cross_val_result_2[0]:.4f}")
    print(f"F1: {cross_val_result_1[1]:.4f} - {cross_val_result_2[1]:.4f}")
    print(f"Precision: {cross_val_result_1[2]:.4f} - {cross_val_result_2[2]:.4f}")
    print(f"Recall: {cross_val_result_1[3]:.4f} - {cross_val_result_2[3]:.4f}")
    print(f"Kappa: {cross_val_result_1[4]:.4f} - {cross_val_result_2[4]:.4f}")
    print(f"\nConfusion Matrix for {model_1_name}:")
    print_cm_simple(cross_val_result_1[5])
    print(f"\nConfusion Matrix for {model_2_name}:")
    print_cm_simple(cross_val_result_2[5])

def print_results(model, features, labels):
    cross_validation_result = cross_validation_one_model(model, features, labels)
    print(f"Accuracy:  {cross_validation_result[0]:.4f}")
    print(f"F1:  {cross_validation_result[1]:.4f}")
    print(f"Precision: {cross_validation_result[2]:.4f}")
    print(f"Recall: {cross_validation_result[3]:.4f}")
    print(f"Kappa: {cross_validation_result[4]:.4f}")
    print_cm_simple(cross_validation_result[5])

"""
DONT FORGET TO SAY THIS IS AI!!! "Can you add labels to this confusion matrix" Claude

Changed some variable names for clarity
"""


def print_cm_simple(confusion_matrix):
    """Simple confusion matrix with clear axes"""
    labels = ['fake', 'sat', 'bias', 'cons', 'junk',
              'hate', 'click', 'unrel', 'pol', 'rel']

    print("Rows = Actual | Columns = Predicted\n")

    # Header
    print("        ", end="")
    for label in labels:
        print(f"{label:>6}", end="")
    print()

    # Separator
    print("      " + "-" * 66)

    # Rows
    for index, row in enumerate(confusion_matrix):
        print(f"{labels[index]:>6} |", end="")
        for value in row:
            print(f"{value:>6}", end="")
        print()
