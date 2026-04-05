"""
Filename: sampling.py
Author: Beck Chamberlain
Version: 0.05
Description: This script performs sampling on the 'news_sample.csv' file.
"""
import csv
import pandas as pd
import random
import sys


def reservoir_sample_from_csv(number_of_desired_rows):
    """
    Returns a sample from the 'news_sample.csv' file with equal parts of each selected classifier.

    This is done by creating a reservoir for each class going over the csv in chunks and populating the reservoir
    using reservoir sampling techniques.
    """
    # Sets how many rows there will be for every class
    number_of_rows_per_class = number_of_desired_rows // 10
    # Increases the maximum CSV field size so large body's don't create errors
    csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
    # Stores samples for each class
    classifier_labels = ['fake', 'satire', 'bias', 'conspiracy', 'junksci', 'hate', 'clickbait', 'unreliable',
                         'political', 'reliable']
    # Stores max reservoir size for each class
    class_max_reservoir = {label: number_of_rows_per_class for label in classifier_labels}
    class_max_reservoir['bias'] = number_of_rows_per_class + number_of_rows_per_class * 0.1
    # Initializes reservoirs for each class
    reservoirs = {class_type: [] for class_type in classifier_labels}
    number_of_entries_per_class = {class_type: 0 for class_type in classifier_labels}
    # Chosen fixed random seed
    random.seed(9)

    # Goes through the fake news csv in chunks to avoid overflow, uses single row dataframe to ensure the header is kept
    for chunk in pd.read_csv('data/raw/archive/fake_news.csv', chunksize=10000, engine='python', on_bad_lines='skip',
                             index_col=0):
        # for each row in the chunk
        for _, row in chunk.iterrows():
            # Gets the classification type of the current row
            class_label = row['type']
            # Skips any unlabeled entries or rogue entries
            if class_label not in reservoirs:
                continue
            # Appends number of entries for the current rows class
            number_of_entries_per_class[class_label] += 1
            # If the number of entries for that class needed for the sample has not been reached it is added
            if len(reservoirs[class_label]) < class_max_reservoir[class_label]:
                reservoirs[class_label].append(row.to_frame().T)
            else:
                # Finds a random position between 0 and the number of records of the current class
                random_position_inside_reservoir = random.randint(0, number_of_entries_per_class[class_label] - 1)
                # If the random position is less than the size of the sample then it is replaced then the current row
                # replaces the row in that position
                if random_position_inside_reservoir < number_of_rows_per_class:
                    reservoirs[class_label][random_position_inside_reservoir] = row.to_frame().T

    # Combine all samples into a singular csv by concatenating all the reservoirs into a panda then converting it to a csv
    pandas_sample = pd.concat([pd.concat(rows, ignore_index=True)
                               for rows in reservoirs.values()], ignore_index=True)
    pandas_sample.to_csv(f"data/samples/news_sample_{number_of_desired_rows}.csv", index=False)


def random_sample_from_csv(number_of_desired_rows):
    """
    Returns a random sample of a given size from the 'news_sample.csv' file.

    This is done by going through the file row by row and using reservoir sampling techniques.
    """
    # Increases the maximum CSV field size so large body's don't create errors
    csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
    # Path to the fake news dataset
    fake_news_csv_filename = "data/raw/archive/fake_news.csv"
    # Path to where the sample of the dataset will be saved
    random_sample_filename = f"data/samples/random_news_sample_{number_of_desired_rows}.csv"
    # Chosen fixed random seed
    random.seed(3)

    # Open the fake news dataset and takes a random selection of the rows
    with open(fake_news_csv_filename, encoding="utf-8", errors="replace", newline="") as news_csv_file:
        news_csv_reader = csv.reader(news_csv_file)
        # Skips and saves the header
        news_csv_header = next(news_csv_reader)

        csv_rows = []  # Holds the rows that make up the sample
        # Adds the number of desired rows and then randomly replaces these rows
        for index, row in enumerate(news_csv_reader):
            if index < number_of_desired_rows:
                # Adds the number of desired rows
                csv_rows.append(row)
            else:
                # Selects a row in the dataset between the first and the current index
                random_row_selection = random.randint(0, index)
                # If the index is below the number of desired rows the record associated
                if random_row_selection < number_of_desired_rows:
                    csv_rows[random_row_selection] = row

    # Creates a csv file and writes the chosen rows to it as well as the previously saved header
    with open(random_sample_filename, "w", encoding="utf-8", newline="") as sample_csv_file:
        writer = csv.writer(sample_csv_file)
        writer.writerow(news_csv_header)
        writer.writerows(csv_rows)
