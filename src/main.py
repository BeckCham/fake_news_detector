"""

"""
import csv
import pandas as pd

from src.models import classifiers
from src.preprocessing import csv_cleaning, added_features
from src.ingestion.web_scraping import url_to_data
from src.preprocessing.text_representations import tf_idf
#from src.ui.gui import run_gui

#if __name__ == "__main__":
# run_gui()

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

# csv_analysis.size_of_dataframe('data/samples/news_sample_10000.csv')

# csv_analysis.number_of_missing_data('data/cleaned/cleaned_sample_1.csv')
# csv_analysis.number_of_non_english_articles('data/samples/news_sample_10000_2.csv')
# csv_analysis.class_balance("data/cleaned/cleaned.csv")
"""
Gets data from URL
"""
# url_to_data('https://www.pravda.ru/world/1139645-negr/')
"""
Applies tf-idf to cleaned news sample
"""
tf_idf.create_tfidf_dataset('data/cleaned/cleaned_sample_1.csv')
"""
Runs NB
"""
from src.models.classifiers import train_model

#classifiers.train_model("naive_bayes", 1)
#classifiers.run_naive_bayes()
"""
Runs Decision Tree
"""
# from src.models.decision_tree import decision_trees
# decision_trees.run_decision_tree()

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