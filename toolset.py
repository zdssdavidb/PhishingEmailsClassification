# Toolset for Detecting Phishing emails using ML.
# Date started: 09.03.2023

import gensim, os, pandas, numpy, pickle, re, datetime, glob, time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, plot_confusion_matrix
from sklearn.preprocessing import LabelBinarizer

# Opening dataset, splitting + PCA
def get_features_and_labels_my_code(PCA_limit, training_dataset, training_dataset_label, evaluation_dataset, evaluation_dataset_label):
    df = pandas.read_csv(training_dataset)  # training dataset
    features = numpy.array(df)[:,1:-1]      # getting rid of index column too
    labels = numpy.array(df)[:,-1]
    # Splitting into training/testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, shuffle=True)        # Shuffle applied
    print(f"Number of spam emails in training set: {numpy.count_nonzero(train_labels, axis=None)}")       # check how many spam labels in training labels
    print(f"Number of spam emails in testing set: {numpy.count_nonzero(test_labels, axis=None)}")        # check how many spam labels in testing labels
    train_features, test_features = pca_feature_reduction(train_features, test_features, PCA_limit)         # Applying PCA reduction to training data 
    # Evaluation dataset
    df2 = pandas.read_csv(evaluation_dataset)
    eval_features = numpy.array(df2)[:,1:-1]      # getting rid of index column too
    eval_labels = numpy.array(df2)[:,-1]
    eval_train_features, eval_test_features, eval_train_labels, eval_test_labels  = train_test_split(eval_features, eval_labels, test_size=0.2, shuffle=True)
    eval_train_features, eval_test_features = pca_feature_reduction(eval_train_features, eval_test_features, PCA_limit)   # Applying PCA reduction to testing data
    print(f"Number of spam emails in training set: {numpy.count_nonzero(train_labels, axis=None)}")       # check how many spam labels in training labels
    print(f"Number of spam emails in testing set: {numpy.count_nonzero(test_labels, axis=None)}")        # check how many spam labels in testing labels
    return features, labels, train_features, test_features, train_labels, test_labels, eval_features, eval_labels, eval_train_features, eval_test_features, eval_train_labels, eval_test_labels

# Opening dataset, splitting NO PCA
def get_features_and_labels_phish1(PCA_limit,training_dataset, training_dataset_label, evaluation_dataset, evaluation_dataset_label):
    df = pandas.read_csv(training_dataset)  # training dataset
    features = numpy.array(df)[:,1:-1]      # getting rid of index column too
    labels = numpy.array(df)[:,-1]
    encoder = LabelBinarizer()
    labels = encoder.fit_transform(labels.ravel())  # phish1 matrix needs it to be this way, apparently.
    # Splitting into training/testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, shuffle=True)        # Shuffle applied
    print(f"Number of spam emails in training set: {numpy.count_nonzero(train_labels, axis=None)}")       # check how many spam labels in training labels
    print(f"Number of spam emails in testing set: {numpy.count_nonzero(test_labels, axis=None)}")        # check how many spam labels in testing labels
    # Evaluation dataset
    df2 = pandas.read_csv(evaluation_dataset)
    eval_features = numpy.array(df2)[:,1:-1]      # getting rid of index column too
    eval_labels = numpy.array(df2)[:,-1]
    eval_labels = encoder.transform(eval_labels)    # encoding eval_labels
    eval_train_features, eval_test_features, eval_train_labels, eval_test_labels  = train_test_split(eval_features, eval_labels, test_size=0.2, shuffle=True)
    print(f"Number of spam emails in training set: {numpy.count_nonzero(train_labels, axis=None)}")       # check how many spam labels in training labels
    print(f"Number of spam emails in testing set: {numpy.count_nonzero(test_labels, axis=None)}")        # check how many spam labels in testing labels
    return features, labels, train_features, test_features, train_labels, test_labels, eval_features, eval_labels, eval_train_features, eval_test_features, eval_train_labels, eval_test_labels

def get_features_and_labels_phish3(PCA_limit, training_dataset, training_dataset_label, evaluation_dataset, evaluation_dataset_label):
    # Opening dataset, splitting and normalizing if necessary
    df = pandas.read_csv(training_dataset)  # training dataset
    features = numpy.array(df)[:,1:-1].astype('float64')      # getting rid of index column too
    labels = numpy.array(df)[:,-1].astype('int64')
    # Splitting into training/testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, shuffle=True)        # Shuffle applied
    print(f"Number of spam emails in training set: {numpy.count_nonzero(train_labels, axis=None)}")       # check how many spam labels in training labels
    print(f"Number of spam emails in testing set: {numpy.count_nonzero(test_labels, axis=None)}")        # check how many spam labels in testing labels
    train_features, test_features = pca_feature_reduction(train_features, test_features, PCA_limit)         # Applying PCA reduction to training data 
    # Evaluation
    df2 = pandas.read_csv(evaluation_dataset)
    eval_features = numpy.array(df2)[:,1:-1]      # getting rid of index column too
    eval_labels = numpy.array(df2)[:,-1]
    eval_train_features, eval_test_features, eval_train_labels, eval_test_labels  = train_test_split(eval_features, eval_labels, test_size=0.2, shuffle=True)
    eval_train_features, eval_test_features = pca_feature_reduction(eval_train_features, eval_test_features, PCA_limit)   # Applying PCA reduction to testing data
    print(f"Number of spam emails in training set: {numpy.count_nonzero(train_labels, axis=None)}")       # check how many spam labels in training labels
    print(f"Number of spam emails in testing set: {numpy.count_nonzero(test_labels, axis=None)}")        # check how many spam labels in testing labels
    return features, labels, train_features, test_features, train_labels, test_labels, eval_features, eval_labels, eval_train_features, eval_test_features, eval_train_labels, eval_test_labels

def evaluation(test_labels, predictions, labels, classifier, test_features):
    # Measuring performance
    from sklearn.metrics import confusion_matrix
    confusion_matrix(test_labels, predictions)
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    # print results in human
    print(f"Accuracy:{accuracy}")
    print(f"Precision:{precision}")
    print(f"Recall:{recall}")
    print(f"F1 Score:{f1}")

    # # Measuring TP, TN, FP, FN (not finished)
    # def perf_measure(true_labels, predicted_labels):
    #     TP = 0
    #     FP = 0
    #     TN = 0
    #     FN = 0
    #     for i in range(len(predicted_labels)): 
    #         if true_labels[i]==predicted_labels[i]==1:
    #             TP += 1
    #         elif predicted_labels[i]==1 and true_labels[i]!=predicted_labels[i]:
    #             FP += 1
    #         elif true_labels[i]==predicted_labels[i]==0:
    #             TN += 1
    #         elif predicted_labels[i]==0 and true_labels[i]!=predicted_labels[i]:
    #             FN += 1
    #         else:
    #             print(f"\nPredicted:{predicted_labels[i]}, actual:{true_labels[i]}\n")
    #     return(TP,TN,FP,FN)
    # conf_matr = perf_measure(labels,predictions)
    # TP=conf_matr[0]
    # TN=conf_matr[1]
    # FP=conf_matr[2]
    # FN=conf_matr[3]
    # return accuracy, precision, recall, f1, TP, TN, FP, FN
    return accuracy, precision, recall, f1

# Saving trained model (not tested)
def save_model(model, model_save_filename_FULL):
    # Saving models
    os.chdir("D:\Download")
    for classifier in classifiers:
        name = str(type(classifier)).split(".")[-1][:-2]
        f = open(f"{name}.pkl", 'wb')
        pickle.dump(classifier, f)
        f.close()
        print("\nModel Saved")

# PCA feature reduction (optional)
def pca_feature_reduction(train_features, test_features, PCA_limit):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=PCA_limit)
    train_features = pca.fit_transform(train_features)
    test_features = pca.transform(test_features)
    return train_features, test_features

# Saving results to stats_file provided when calling the function.
def save_stats(model_name, PCA_limit, accuracy, precision, recall, f1, elapsed_time, labels, stats_file):
    import os
    # model_name = str(i).split(" ")[0].split('(')[0]
    os.system(f"echo {model_name},{PCA_limit},{accuracy},{precision},{recall},{f1},{elapsed_time},{labels.shape[0]} >>{stats_file}") # saving training data to csv

# this uses TP, TN, FP and FN as well.
def save_stats_full(model_name, PCA_limit, accuracy, precision, recall, f1, elapsed_time, labels, stats_file):
    import os
    # model_name = str(i).split(" ")[0].split('(')[0]
    os.system(f"echo {model_name},{PCA_limit},{accuracy},{precision},{recall},{f1},{elapsed_time},{labels.shape[0]},{TP},{TN},{FP},{FN} >>{stats_file}") # saving training data to csv

# Classifying with an array of ML classifiers, going through training dataset first and then evaluation dataset.
# this one does not include SVM Classifier as it doesn't work with this matrix. 
def classification_phish1(features, labels, train_features, train_labels, test_features, test_labels, stats_file, PCA_limit, training_dataset, training_dataset_label, evaluation_dataset, evaluation_dataset_label, eval_train_features, eval_train_labels, eval_test_features, eval_labels, eval_test_labels, switch):
    # building the bunch
    import os
    if switch == 1:
        print("Initializing ML Classifiers, fitting each one.")
        t = time.process_time()
        classifiers = []
        from sklearn.tree import DecisionTreeClassifier	
        DT= DecisionTreeClassifier()
        classifiers.append(DT)
        from sklearn.linear_model import LogisticRegression
        LR = LogisticRegression(random_state=0, class_weight='balanced')
        classifiers.append(LR)
        from sklearn.linear_model import Perceptron	
        perceptron = Perceptron(verbose=False)
        classifiers.append(perceptron)
        from sklearn.ensemble import RandomForestClassifier
        RF = RandomForestClassifier(n_estimators=100)
        classifiers.append(RF)
        from sklearn.naive_bayes import GaussianNB
        NB = GaussianNB()
        classifiers.append(NB)
        from sklearn.ensemble import AdaBoostClassifier
        adaboost = AdaBoostClassifier(n_estimators=100, random_state=0)
        classifiers.append(adaboost)
        from sklearn.neural_network import MLPClassifier
        MLP = MLPClassifier( solver='lbfgs', activation='logistic', early_stopping=True, max_iter=2000)
        classifiers.append(MLP)
        from sklearn.ensemble import HistGradientBoostingClassifier
        HGBC = HistGradientBoostingClassifier()
        classifiers.append(HGBC)
        from sklearn.ensemble import GradientBoostingClassifier
        GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
        classifiers.append(GBC)
        print("Starting group classification")
        # Using each ML model, measuring accuracy and evaluating, saving both stats to stats_file
        for i, model in enumerate (classifiers):
            try:
                model_name = str(type(model)).split(" ")[1].split('_')[1].split(".")[1][:-2]        # extract model name from class name
                if len(model_name)<1:
                    model_name = str(type(model)).split(".")[3][:-2]
            except IndexError:
                model_name = str(type(model)).split(".")[3][:-2]
            except:
                model_name = str(type(model))
            print(f"\nTraining {model_name}\n")
            model.fit(train_features, train_labels.ravel())
            predictions = model.predict(test_features)
            # Measure accuracy and save stats to CSV
            accuracy, precision, recall, f1 = evaluation(test_labels, predictions, labels, type(model), test_features)   # getting f1, accuracy, etc
            elapsed_time = (time.process_time() - t)/60		# time took in minutes
            save_stats(model_name, PCA_limit, accuracy, precision, recall, f1, elapsed_time, labels, stats_file)
            # Evaluating with testing compilation and Saving stats to CSV
            t = time.process_time() 
            # print(f"\nEvaluating {type(model)}\n")
            print(f"\nEvaluating {model_name}\n") 
            
            # With re-fitting for testing dataset
            model.fit(eval_train_features, eval_train_labels.ravel())
            eval_predictions = model.predict(eval_test_features)
            elapsed_time = (time.process_time() - t)/60		# time took in minutes
            accuracy, precision, recall, f1 = evaluation(eval_test_labels, eval_predictions, eval_labels, type(model), eval_test_features)  # getting f1, accuracy, etc
            save_stats(model_name, PCA_limit, accuracy, precision, recall, f1, elapsed_time, eval_labels, stats_file)
            
            # # Without re-fitting for testing dataset (different dataset, different samples)
            # eval_predictions = model.predict(eval_test_features)
            # elapsed_time = (time.process_time() - t)/60		# time took in minutes
            # accuracy, precision, recall, f1 = evaluation(eval_test_labels, eval_predictions, eval_labels, type(model), eval_test_features)  # getting f1, accuracy, etc
            # save_stats(model_name, PCA_limit, accuracy, precision, recall, f1, elapsed_time, eval_labels, stats_file)
            
            # plot_confusion_matrix(model, test_features, test_labels) # plot Confusion Matrix (training)
            # plot_confusion_matrix(model, eval_test_features, eval_test_labels) # plot Confusion Matrix (evaluation)

            # # Saving models
            # os.chdir("D:\Download")
            # name = str(type(model)).split(".")[-1][:-2]
            # f = open(f"{name}.pkl", 'wb')
            # pickle.dump(model, f)
            # f.close()
            # print(f"\n{name} Saved")

    # using pre-trained models    
    elif switch == 2:
        print("Initializing ML Classifiers, fitting each one.")
        t = time.process_time()
        classifiers = []
        from sklearn.tree import DecisionTreeClassifier	
        DT= DecisionTreeClassifier()
        classifiers.append(DT)
        from sklearn.linear_model import LogisticRegression
        LR = LogisticRegression(random_state=0, class_weight='balanced')
        classifiers.append(LR)
        from sklearn.linear_model import Perceptron	
        perceptron = Perceptron(verbose=True)
        classifiers.append(perceptron)
        from sklearn.ensemble import RandomForestClassifier
        RF = RandomForestClassifier(n_estimators=100)
        classifiers.append(RF)
        from sklearn.naive_bayes import GaussianNB
        NB = GaussianNB()
        classifiers.append(NB)
        from sklearn.ensemble import AdaBoostClassifier
        adaboost = AdaBoostClassifier(n_estimators=100, random_state=0)
        classifiers.append(adaboost)
        from sklearn.neural_network import MLPClassifier
        MLP = MLPClassifier( solver='lbfgs', activation='logistic', early_stopping=True)
        classifiers.append(MLP)
        from sklearn.ensemble import HistGradientBoostingClassifier
        HGBC = HistGradientBoostingClassifier()
        classifiers.append(HGBC)

        print("Starting group classification")
        # Using each ML model, measuring accuracy and evaluating, saving both stats to stats_
        for i, model in enumerate (classifiers):
            try:
                model_name = str(type(model)).split(" ")[1].split('_')[1].split(".")[1][:-2]        # extract model name from class name
                if len(model_name)<1:
                    model_name = str(type(model)).split(".")[3][:-2]
            except IndexError:
                model_name = str(type(model)).split(".")[3][:-2]
            except:
                model_name = str(type(model))
            print(f"\nTraining {model_name}\n")
            model.fit(train_features, train_labels.ravel())
            predictions = model.predict(test_features)
            # Measure accuracy and save stats to CSV
            accuracy, precision, recall, f1 = evaluation(test_labels, predictions, labels, type(model), test_features)   # getting f1, accuracy, etc
            elapsed_time = (time.process_time() - t)/60		# time took in minutes
            save_stats(model_name, PCA_limit, accuracy, precision, recall, f1, elapsed_time, labels, stats_file)
            # Evaluating with testing compilation and Saving stats to CSV
            t = time.process_time() 
            # print(f"\nEvaluating {type(model)}\n")
            print(f"\nEvaluating {model_name}\n")
            model = model.fit(eval_train_features, eval_train_labels.ravel())
            eval_predictions = model.predict(eval_test_features)
            elapsed_time = (time.process_time() - t)/60		# time took in minutes
            accuracy, precision, recall, f1 = evaluation(eval_test_labels, eval_predictions, eval_labels, type(model), eval_test_features)  # getting f1, accuracy, etc

            save_stats(model_name, PCA_limit, accuracy, precision, recall, f1, elapsed_time, eval_labels, stats_file)
    
# Classifying, applying PCA reduction (if specified)
def classification_my_code(features, labels, train_features, train_labels, test_features, test_labels, stats_file, PCA_limit, training_dataset, training_dataset_label, evaluation_dataset, evaluation_dataset_label, eval_train_features, eval_train_labels, eval_test_features, eval_labels, eval_test_labels): 
    # building the bunch
    import os
    print("Initializing ML Classifiers, fitting each one.")
    t = time.process_time()
    classifiers = []
    from sklearn.svm import SVC
    SVM = SVC(kernel='linear', probability=True)
    classifiers.append(SVM)
    from sklearn.tree import DecisionTreeClassifier	
    DT= DecisionTreeClassifier()
    classifiers.append(DT)
    from sklearn.linear_model import LogisticRegression
    LR = LogisticRegression(random_state=0, class_weight='balanced')
    classifiers.append(LR)
    from sklearn.linear_model import Perceptron	
    perceptron = Perceptron(verbose=False)
    classifiers.append(perceptron)
    from sklearn.ensemble import RandomForestClassifier
    RF = RandomForestClassifier(n_estimators=100)
    classifiers.append(RF)
    from sklearn.naive_bayes import GaussianNB
    NB = GaussianNB()
    classifiers.append(NB)
    from sklearn.ensemble import AdaBoostClassifier
    adaboost = AdaBoostClassifier(n_estimators=100, random_state=0)
    classifiers.append(adaboost)
    from sklearn.neural_network import MLPClassifier
    MLP = MLPClassifier( solver='lbfgs', activation='logistic', early_stopping=True, max_iter=10000)
    classifiers.append(MLP)
    from sklearn.ensemble import HistGradientBoostingClassifier
    HGBC = HistGradientBoostingClassifier()
    classifiers.append(HGBC)
    from sklearn.ensemble import GradientBoostingClassifier
    GBC = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    classifiers.append(GBC)
    print("Starting group classification")

    # Using each ML model, measuring accuracy and evaluating, saving both stats to stats_
    for i, model in enumerate (classifiers):
        try:
            model_name = str(type(model)).split(" ")[1].split('_')[1].split(".")[1][:-2]        # extract model name from class name
            if len(model_name)<1:
                model_name = str(type(model)).split(".")[3][:-2]
        except IndexError:
            model_name = str(type(model)).split(".")[3][:-2]
        except:
            model_name = str(type(model))
        print(f"\nTraining {model_name}\n")
        model.fit(train_features, train_labels)
        predictions = model.predict(test_features)
        # Measure accuracy and save stats to CSV
        accuracy, precision, recall, f1 = evaluation(test_labels, predictions, labels, type(model), test_features)   # getting f1, accuracy, etc
        elapsed_time = (time.process_time() - t)/60		# time took in minutes
        save_stats(model_name, PCA_limit, accuracy, precision, recall, f1, elapsed_time, labels, stats_file)
        # Evaluating with testing compilation and Saving stats to CSV
        t = time.process_time() 
        # print(f"\nEvaluating {type(model)}\n")
        print(f"\nEvaluating {model_name}\n")
        model = model.fit(eval_train_features, eval_train_labels)
        eval_predictions = model.predict(eval_test_features)
        elapsed_time = (time.process_time() - t)/60		# time took in minutes
        accuracy, precision, recall, f1 = evaluation(eval_test_labels, eval_predictions, eval_labels, type(model), eval_test_features)  # getting f1, accuracy, etc
        save_stats(model_name, PCA_limit, accuracy, precision, recall, f1, elapsed_time, eval_labels, stats_file)
        plot_confusion_matrix(model, test_features, test_labels) # plot Confusion Matrix (training)
        plot_confusion_matrix(model, eval_test_features, eval_test_labels) # plot Confusion Matrix (evaluation) 

        # # Saving model
        # os.chdir("D:\Download")
        # name = str(type(model)).split(".")[-1][:-2]
        # f = open(f"{name}.pkl", 'wb')
        # pickle.dump(model, f)
        # f.close()
        # print(f"\n{name} Saved")

