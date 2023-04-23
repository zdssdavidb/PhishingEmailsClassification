# PhishingEmailsClassification
Classifying Phishing emails with Machine Learning and DL.
For Honours Project (2023).

1. Extract features from a folder of emails (.eml, .txt work fine).

file method1_feature_extraction_test_emails.py opens a specified folder and processes all emails found,then adding a specified label and outputting all to the specified csv file.
file method1_feature_extraction.py opens 2 folders and applies different labels (benign and phishing) and out produces a combined dataset of phishing/benign with labels.
file method2_feature_extraction.py does the same as above but without adding labels and does not combine multiple datasets, only opens 1 at a time (so far).

2. Combine phishing & genuine emails features in 1 dataset.
3. Classify using the classification functions in the toolbox (contains the following ML classifiers: SVM Classifier, Decision Tree, LogisticRegression, Perceptron, Random Forest, Gaussian NaiveBayes, AdaBoost, MLP classifier, Hist Gradient Boosting Classifier, Gradient Boosting Classifier). [ this is ran through a loop of x repetitions, so can automatically re-run, say 4x times if "4" is specified.
4. Analyse classification accuracy, precision, recall, f1 score in output CSV files.

file "method1_classification.py" takes a combined dataset with labels (location specified in the code) and opens, separates features and labels and trains all classifiers and gets predictions, predictions are then saved to a specified output CSV file. PCA is applied if the "PCA_limit" variable is reduced from 40 for Method 1 and 91 for Method 2. Number of iterations (number of accuracy checks) can be defined with the "iterations" variable.
file "method2_classification.py" does exactly the same but for datasets compiled using Method 2 of feature extraction. Also has SVM classifier working, unlike the file above.

NB. Feature extractions have some differences and separate functions for opening, splitting and classification are present in the toolset file.
