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
from src.preprocessing import csv_cleaning, added_features
from src.ingestion.web_scraping import url_to_data
from src.preprocessing.lfr import create_tfidf_dataset

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

#csv_analysis.size_of_dataframe('data/samples/news_sample_10000.csv')

#csv_analysis.number_of_missing_data('data/samples/news_sample_10000_1.csv')
# csv_analysis.number_of_non_english_articles('data/samples/news_sample_10000_2.csv')
# csv_analysis.class_balance("data/cleaned/cleaned.csv")
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
from src.preprocessing.lfr import create_tfidf_dataset,prepare_data
#create_tfidf_dataset(*prepare_data('data/cleaned/cleaned_sample_1.csv'))
"""
Runs NB
"""
from src.models.classifiers import train_model, grid_search

#classifiers.train_model("naive_bayes", '1',True)
#classifiers.run_naive_bayes()
"""
Runs Decision Tree
"""
#classifiers.train_model("decision_tree", '1',True)
"""
Runs Random Forest
"""
#
#classifiers.train_model("random_forest", '1',True)
"""
Runs KNN
"""
#classifiers.train_model("knn", '1',True)
"""
Runs SVM
"""
#classifiers.train_model("svm", '1',True)
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
Does Wilcoxen tests between two different models
"""
from src.models.model_comparison import wilcoxon_between_means, cross_validation_one_model
#t_test_between_samples(77.42,77.51,"sample 1","sample 2")
"""
Performs feature selection
"""
from src.preprocessing.lfr import prepare_data
from src.preprocessing.feature_selection import test_features_individually,find_best_combination_of_features, check_best_combination,tf_idf_feature_selection_with_varying_features,tf_idf_print_accuracies_with_change
#test_features_individually(*prepare_data('data/cleaned/cleaned_sample.csv')[:3])
#find_best_combination_of_features(*prepare_data('data/cleaned/cleaned_sample.csv')[:3])
#check_best_combination(*prepare_data('data/cleaned/cleaned_sample.csv')[:3])
#prepared_data = prepare_data('data/cleaned/cleaned_sample_1.csv')
#tf_idf_feature_selection_with_varying_features(prepared_data[0], prepared_data[2],"svm")
#tf_idf_print_accuracies_with_change(prepared_data[0], prepared_data[2])
"""
Find best hyperparameters
"""
#grid_search('svm', 1)
"""
Check if models are significantly diffrent
"""

"""
Train voting model
"""
#classifiers.train_model("voting", '1',True)