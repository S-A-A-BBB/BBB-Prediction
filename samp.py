# oversample will multiple classes (minority class)
from imblearn.over_sampling import SMOTE
import imblearn
import pandas as pd

print(imblearn.__version__)
import sys

sys.path.append('..')

dataset_orig = pd.read_csv('Dataset/ORGINAL DATASET WITH FINGERPRINTS.csv')

labels = dataset_orig['Labels'].values
dataset = dataset_orig.drop(columns=['Labels'])

# we are coverting from dataframe into an numpy array
dataset_sampled = dataset.iloc[:, :]

sampler = SMOTE(k_neighbors=12)

dataset_sampled, labels_resampled = sampler.fit_sample(dataset_sampled, labels)

def smote():
    return dataset_sampled, labels_resampled
