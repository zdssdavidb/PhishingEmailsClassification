# Classifying training and evaluation datasets with a bunch of ML classifiers (defined in the toolset library). This is for the my_code features matrix (91 Word2Vec vectors).

import os, time, seaborn, pandas
os.chdir(r"C:\Users\Cube\OneDrive - GLASGOW CALEDONIAN UNIVERSITY\Honours Project\Year 4")
from toolset import *


# My code
stats_file = r"D:\Download\method2_classification_output.csv"
training_dataset = r"C:\Users\Cube\OneDrive - GLASGOW CALEDONIAN UNIVERSITY\Honours Project\Year 4\my_code\matrix\combined_dataset_balanced_2000.csv"
training_dataset_label = "combined_dataset_balanced_2000.csv"
evaluation_dataset = r"C:\Users\Cube\OneDrive - GLASGOW CALEDONIAN UNIVERSITY\Honours Project\Year 4\my_code\matrix\testing_compilation_v1.csv"
evaluation_dataset_label = "testing_compilation_v1.csv"
PCA_limit = 91
iterations = 4
# 1. Open matrix dataset and split
features, labels, train_features, test_features, train_labels, test_labels, eval_features, eval_labels, eval_train_features, eval_test_features, eval_train_labels, eval_test_labels = get_features_and_labels_my_code(PCA_limit, training_dataset, training_dataset_label, evaluation_dataset, evaluation_dataset_label)

# loop through iterations of classification and accuracy check
for iteration in range(1,iterations+1):
    # 2. run the selected csv's through 
    classification_my_code(features, labels, train_features, train_labels, test_features, test_labels, stats_file, PCA_limit, training_dataset, training_dataset_label, evaluation_dataset, evaluation_dataset_label, eval_train_features, eval_train_labels, eval_test_features, eval_labels, eval_test_labels)

