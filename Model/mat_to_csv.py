#!/Users/nico/Desktop/GaitUP - UP/mat_to_csv.py
import os
import re
import csv
import logging
import pandas as pd
from scipy.io import loadmat


class MatToCsv:
    def dataset(self, file):

        # Setting the matlab variables
        mat_file = file
        mat_data = loadmat(mat_file)

        # CSV TEMPORAL
        csv_filename = os.path.basename(mat_file) + '.csv'
        open(csv_filename, "w").close()

        # Checking matlab columns
        for col in mat_data.keys():
            # Only process the non-administrative column (non __)
            if not re.search("__", col):
                with open(csv_filename, 'w', newline='') as csv_file:
                    logging.info('Processing {} converting into {}'.format(mat_file, csv_filename))
                    writer = csv.writer(csv_file)
                    subjects = []
                    for subject_id in range(0, 21):
                        subjects.append("subject_{}".format(subject_id))
                    writer.writerow(subjects)
                    for row in mat_data[col]:
                        writer.writerow(row)
        dataset = pd.read_csv(csv_filename)


        os.remove(csv_filename)  #remove if want to delete
        return dataset
