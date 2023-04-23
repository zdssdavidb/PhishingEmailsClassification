# Classifying training and evaluation datasets with a bunch of ML classifiers (defined in the toolset library).

import os, time, pandas
os.chdir(r"C:\Users\Cube\OneDrive - GLASGOW CALEDONIAN UNIVERSITY\Honours Project\Year 4")
from toolset import *

# Phish1
stats_file = r"D:\Download\method1_classification_output.csv"
training_dataset = r"D:\Download\method1_combined_dataset.csv"
training_dataset_label = "method1_combined_dataset.csv"
evaluation_dataset = r"C:\Users\Cube\OneDrive - GLASGOW CALEDONIAN UNIVERSITY\Honours Project\Year 4\Datasets\matrix\phish1\testing_compilation_python3.csv"
evaluation_dataset_label = "testing_compilation_python3.csv"
PCA_limit = 40          # reduce amount of features from 40 to...
iterations = 2
switch = 1          # 1 if training classifiers from scratch; 2 if using pre-trained
# 1. Open matrix dataset and split
features, labels, train_features, test_features, train_labels, test_labels, eval_features, eval_labels, eval_train_features, eval_test_features, eval_train_labels, eval_test_labels = get_features_and_labels_phish1(PCA_limit, training_dataset, training_dataset_label, evaluation_dataset, evaluation_dataset_label)


# loop through iterations of classification and accuracy check
for iteration in range(1,iterations+1):
    # 2. run the selected csv's through 
    classification_phish1(features, labels, train_features, train_labels, test_features, test_labels, stats_file, PCA_limit, training_dataset, training_dataset_label, evaluation_dataset, evaluation_dataset_label, eval_train_features, eval_train_labels, eval_test_features, eval_labels, eval_test_labels, switch)


