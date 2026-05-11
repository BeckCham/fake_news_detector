"""
Filename: main.py
Author: Beck Chamberlain
Version: 0.07
Description:
"""
import csv
import pickle

import numpy as np
import pandas as pd
from numpy.ma.core import get_data

from src.models import classifiers
from src.models.model_comparison import wilcoxon_between_models, wilcoxon_between_models_via_identifiers
from src.preprocessing import csv_cleaning, added_features
from src.ingestion.web_scraping import url_to_data
from src.preprocessing.lfr import create_tf_idf_dataset

#from src.ui.gui import run_gui


#if __name__ == "__main__":
    #run_gui()


"""
Preprocesses the csv file given
"""
#csv_cleaning.preprocess_csv('data/samples/news_sample_10000_1.csv')
"""
Gets Sample from news dataset
"""
from ingestion import sampling

# sampling.reservoir_sample_from_csv(10000)

"""
Analyses the csv file given
"""
from src.preprocessing import csv_analysis

#csv_analysis.size_of_dataframe('data/raw/news_sample.csv')

#csv_analysis.dataframe_analysis('data/samples/news_sample_10000_1.csv')
#csv_analysis.number_of_missing_data('data/samples/news_sample_10000_1.csv')
#csv_analysis.number_of_non_english_articles('data/samples/news_sample_10000_2.csv')
#csv_analysis.class_balance("data/samples/news_sample_50000.csv")
"""
Gets data from URL
"""
# url_to_data('https://www.pravda.ru/world/1139645-negr/')
"""
Gets models metrics

with open('src/models/sample_1/decision_tree_model.pkl', 'rb') as file:
    dt_model = pickle.load(file)
features = np.load(f'data/embedded/sample_1/features_tfidf.npy')  # x
labels = np.load(f'data/embedded/sample_1/labels_tfidf.npy')  # y
from src.models.model_comparison import print_results
print_results(dt_model, features, labels)
"""
"""
Applies tf-idf to cleaned news sample
"""
#from src.preprocessing.lfr import create_tf_idf_dataset,prepare_data
#create_tf_idf_dataset(*prepare_data('data/cleaned/cleaned_tf_idf_preprocess.csv'))
"""
Runs NB
"""
from src.models.classifiers import train_model, grid_search

#classifiers.train_model("naive_bayes", 'tf_idf_preprocess_6000',True)
#classifiers.run_naive_bayes()
"""
Runs Decision Tree
"""
#classifiers.train_model("decision_tree", 'tf_idf_preprocess_6000',True)
"""
Runs Random Forest
"""
#classifiers.train_model("random_forest", 'tf_idf_preprocess_6000',True)
"""
Runs KNN
"""
#classifiers.train_model("knn", 'tf_idf_preprocess_6000',True)
"""
Runs SVM
"""
#classifiers.train_model("svm", 'tf_idf_preprocess_6000',True)
"""
Predict from url with nb
"""
# from src.models.naive_bayes import predict_from_url
# print(predict_from_url.predict('https://www.dailymail.co.uk/health/article-15477921/new-superfood-bamboo-heart-inflammation-gut.html'))
"""
Extra features
"""
from src.preprocessing import added_features
#print(added_features.website_credibility_tests("https://www.bbc.co.uk/news/articles/cpwj7r5yxv1o"))

"""
Performs feature selection
"""
from src.preprocessing.lfr import prepare_data
from src.preprocessing.feature_selection import test_features_individually,find_best_combination_of_features, check_best_combination,tf_idf_feature_selection_with_varying_features,tf_idf_print_accuracies_with_change
# Additional features
#test_features_individually(*prepare_data('data/cleaned/cleaned_tf_idf_preprocess.csv')[:3])
#find_best_combination_of_features(*prepare_data('data/cleaned/cleaned_sample.csv')[:3])
#check_best_combination(*prepare_data('data/cleaned/cleaned_sample.csv')[:3])
# TF-IDF
#prepared_data = prepare_data('data/cleaned/cleaned_tf_idf_preprocess.csv')
#tf_idf_feature_selection_with_varying_features(prepared_data[0], prepared_data[2],"knn")
#tf_idf_feature_selection_with_varying_features(prepared_data[0], prepared_data[2],"decision_tree")
#tf_idf_print_accuracies_with_change(prepared_data[0], prepared_data[2])
"""
Find best hyperparameters
"""
#grid_search('naive_bayes', 'tf_idf_preprocess_6000')
#grid_search('decision_tree', 'tf_idf_preprocess_6000')
grid_search('random_forest', 'tf_idf_preprocess_6000')

"""
Check if models are significantly diffrent
"""

"""
Train voting model
"""
#classifiers.train_model("voting", '1',True)
"""
Check results of trained model
"""
#from src.models.model_comparison import print_trained_model_results
#print_trained_model_results('src/models/tf_idf_no_preprocess_5000/naive_bayes_model.pkl','tf_idf_no_preprocess',3000)

"""
Check results between two models
"""
#wilcoxon_between_models_via_identifiers('src/models/tf_idf_no_preprocess_5000/random_forest_model.pkl','tf_idf_no_preprocess_5000',0,'src/models/tf_idf_preprocess_5000/random_forest_model.pkl','tf_idf_preprocess_5000',0)
#wilcoxon_between_models_via_identifiers('src/models/tf_idf_no_preprocess_5000/knn_model.pkl','tf_idf_no_preprocess_5000',0,'src/models/tf_idf_preprocess_5000/knn_model.pkl','tf_idf_preprocess_5000',0)
#wilcoxon_between_models_via_identifiers('src/models/tf_idf_no_preprocess_5000/svm_model.pkl','tf_idf_no_preprocess_5000',0,'src/models/tf_idf_preprocess_5000/svm_model.pkl','tf_idf_preprocess_5000',0)