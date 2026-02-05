import csv
import random
import sys
"""""
Creates a sample of a given size from the fake news dataset using reservoir sampling then saves it.
"""""
def sample_from_csv(number_of_desired_rows):
    #Increases the maximum CSV field size so large body's don't create errors
    csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
    #Path to the fake news dataset
    fake_news_csv = "data/raw/archive/fake_news.csv"
    #Path to where the sample of the dataset will be saved
    sample = f"data/processed/news_sample_{number_of_desired_rows}.csv"
    #Fixed random seed
    random.seed(42)

    #Open the fake news dataset and takes a random selection of the rows
    with open(fake_news_csv,encoding="utf-8", errors="replace", newline="") as news_csv_file:
        news_csv_reader = csv.reader(news_csv_file)
        #Skips and saves the header
        news_csv_header = next(news_csv_reader)


        csv_rows = [] #Holds the rows that make up the sample
        #Adds the number of desired rows and then randomly replaces these rows
        for index, row in enumerate(news_csv_reader):
            if index < number_of_desired_rows:
                #Adds the number of desired rows
                csv_rows.append(row)
            else:
                #Selects a row in the dataset between the first and the current index
                random_row_selection = random.randint(0, index)
                #If the index is below the number of desired rows the record associated
                if random_row_selection < number_of_desired_rows:
                    csv_rows[random_row_selection] = row
    #Creates a csv file and writes the chosen rows to it as well as the previously saved header
    with open(sample, "w",encoding="utf-8", newline="") as sample_csv_file:
        writer = csv.writer(sample_csv_file)
        writer.writerow(news_csv_header)
        writer.writerows(csv_rows)
