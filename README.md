# PhishingEmailsClassification
Classifying Phishing emails with Machine Learning and DL.
For Honours Project (2023).

1. Extract features from a folder of emails (.eml, .txt work fine).
2. Combine phishing & genuine emails features in 1 dataset.
3. Classify using the classification functions in the toolbox (contains the following ML classifiers: SVM Classifier, Decision Tree, LogisticRegression, Perceptron, Random Forest, Gaussian NaiveBayes, AdaBoost, MLP classifier, Hist Gradient Boosting Classifier, Gradient Boosting Classifier). [ this is ran through a loop of x repetitions, so can automatically re-run, say 4x times if "4" is specified.
4. Analyse classification accuracy, precision, recall, f1 score in output CSV files.

NB. Feature extractions have some differences and separate functions for opening, splitting and classification are present in the toolset file.
