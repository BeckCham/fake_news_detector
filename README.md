# Fake News Detector

## Description
A machine learning system that classifies news articles into one of 10 credibility categories. The system consists of two components: an experimentation and training pipeline used to develop and evaluate classification models, and a Tkinter desktop application that accepts a URL and classifies the article using the trained voting ensemble model.

## Features
- Samples a balanced subset of articles from the Fake News Corpus for training.
- Preprocesses and vectorises article text using TF-IDF with n-gram support.
- Extracts additional linguistic and domain-based features including punctuation frequency, lexical diversity, sentence length, pronoun frequency, HTTPS checks, and DMARC verification.
- Trains and evaluates multiple machine learning classifiers: Naïve Bayes, Decision Tree, Random Forest, KNN, and LinearSVC.
- Performs hyperparameter tuning via grid search and statistical significance testing using the Wilcoxon signed-rank test.
- Combines the best performing models into an ensemble voting classifier.
- Provides a desktop application that accepts a URL, scrapes and preprocesses the article, and classifies it using the trained model.
- Displays the classification result in either a simple or detailed format based on user preference.

## Prerequisites
- Python 3.13
- pip

## Installation
Install all dependencies using:
```
pip install -r requirements.txt
```

## Requirements
```
beautifulsoup4==4.14.3
contractions==0.1.73
dnspython==2.8.0
joblib==1.5.3
langdetect==1.0.9
newspaper3k==0.2.8
nltk==3.9.3
numpy==2.4.2
pandas==3.0.0
pyspellchecker==0.9.0
requests==2.25.1
scikit-learn==1.8.0
scipy==1.17.1
```

## Running the Program

### Application
To launch the classification application:
```
python -m src.ui.gui
```

### Training Pipeline
To run the training pipeline, uncomment the relevant functions in `main.py` and run:
```
python main.py
```

## Files

### src/ingestion/sampling.py
Contains functions for sampling a balanced subset of articles from the Fake News Corpus using reservoir sampling.

### src/ingestion/web_scraping.py
Contains functions for scraping article content from a given URL.

### src/preprocessing/csv_preparation.py
Contains functions for cleaning and preparing the dataset, including removing sparse features and merging textual columns.

### src/preprocessing/csv_analysis.py
Contains functions for analysing the dataset, including class balance, missing values, and language distribution.

### src/preprocessing/added_features.py
Contains functions for extracting additional linguistic and domain-based features from articles.

### src/preprocessing/feature_selection.py
Contains functions for applying Chi-Square feature selection to TF-IDF vectors and additional features.

### src/preprocessing/lfr.py
Contains functions for linguistic feature representation used during preprocessing.

### src/training/classifiers.py
Contains functions for training, tuning, and saving machine learning classifiers.

### src/training/model_comparison.py
Contains functions for evaluating and comparing model performance using cross-validation and Wilcoxon significance testing.

### src/ui/gui.py
Entry point for the Tkinter desktop application.

### src/ui/fake_news_detector_gui.py
Contains the Tkinter UI components and layout for the classification application.

### src/ui/predict_from_url.py
Contains functions for processing a URL and generating a classification using the trained model.

### main.py
The main entry point for the training pipeline. Functions can be commented and uncommented to run specific stages of the pipeline.
