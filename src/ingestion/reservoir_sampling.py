import csv

import pandas as pd
import random
import sys
"""
SAY RESERVOIR IS IMPORTANT BECAUSE OF GROUPING OF NEWS SAMPLES AAAAA
"""
def sample_from_csv(number_of_desired_rows):
    # Increases the maximum CSV field size so large body's don't create errors
    csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
    #Stores samples for each class
    types_of_classifier = ['fake','satire','bias','conspiracy','junksci','hate','clickbait','unreliable','political','reliable']
    #Initializes reservoirs for each class
    reservoirs = {class_type: [] for class_type in types_of_classifier}
    number_of_entries_per_class = {class_type: 0 for class_type in types_of_classifier}
    # Chosen fixed random seed
    random.seed(3)

    #Goes through the fake news csv in chunks to avoid overflow
    for chunk in pd.read_csv('data/raw/archive/fake_news.csv', chunksize=10000,engine='python', on_bad_lines='skip'):
        #for each row in the chunk
        for _, row in chunk.iterrows():
            #Gets the classification type of the current row
            class_label = row['type']
            #Skips any unlabeled entries or rogue entries
            if class_label not in reservoirs:
                continue
            #Appends number of entries for the current rows class
            number_of_entries_per_class[class_label] += 1
            #If the number of entries for that class needed for the sample has not been reached it is added
            if len(reservoirs[class_label]) < number_of_desired_rows:
                reservoirs[class_label].append(row)
            else:
                #Finds a random position between 0 and the number of records of the current class
                random_position_inside_reservoir = random.randint(0, number_of_entries_per_class[class_label] - 1)
                #If the random position is less than the size of the sample then it is replaced then the current row
                #replaces the row in that position
                if random_position_inside_reservoir < number_of_desired_rows:
                    reservoirs[class_label][random_position_inside_reservoir] = row

    #Combine all samples into a singular csv by concatenating all the reservoirs into a panda then converting it to a csv
    pandas_sample = pd.concat([pd.DataFrame(row) for row in reservoirs.values()], ignore_index=True)
    pandas_sample.to_csv(f"data/samples/news_sample_{number_of_desired_rows}.csv", index=False)