import pandas as pd
import numpy as np
# Load the dataset
data = pd.read_csv('C:\\Users\\91970\\Downloads\\AI PRC\\hit7\\PRC-7-ADABOOST\\IRIS.csv')
print(data.shape)
print(data.isnull().sum())
print(data.columns)
target_column = 'species'  
y = data[target_column]
x = data.drop(columns=target_column)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


base_classifier = DecisionTreeClassifier(max_depth=3)
adaboost_classifier = AdaBoostClassifier(base_classifier, n_estimators=50, random_state=42)
adaboost_classifier.fit(X_train, y_train)

y_pred = adaboost_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

"""
What is AdaBoost
 AdaBoost short for Adaptive Boosting is an ensemble
 learning used in machine learning for classification and
 regression problems. The main idea behind AdaBoost is to
 iteratively train the weak classifier on the training dataset
 with each successive classifier giving more weightage to
 the data points that are misclassified. The final AdaBoost
 model is decided by combining all the weak classifier that
 has been used for training with the weightage given to the
 models according to their accuracies. The weak model
 which has the highest accuracy is given the highest
 weightage while the model which has the lowest accuracy
 is given a lower weightage.
 Institution Behind AdaBost Algorithm
 ·
 AdaBoost techniques combine many weak machine-learning models to
 create a powerful classification model for the output. The steps to build
 and combine these models are as
 Step1– Initialize the weights
 For a dataset with N training data points instances, initialize N
 ��Wi 
weights for each data point with ��=1� Wi 
=N1 
Step2– Train weak classifiers
 ·
 ·
 Train a weak classifier Mk where k is the current iteration
 The weak classifier we are training should have an accuracy
 greater than 0.5 which means it should be performing better than a
 naive guess
 Step3– Calculate the error rate and importance of each weak model Mk
·
 Calculate rate error_rate for every weak classifier Mk on the
 training dataset
 Calculate the importance of each model α_k using formula
 ·
 ��=12ln 1–������������αk 
=21 
lnerrork 
1–errork 
Step4– Update data point weight for each data point Wi
 ·
 After applying the weak classifier model to the training data we will
 update the weight assigned to the points using the accuracy of the
 model. The formula for updating the weights will be
 ��=��exp (−������(��))wi 
=wi 
exp(−αk 
yi 
Mk 
(xi 
)) . Here yi is the true
 output and Xi is the corresponding input vector
 Step5– Normalize the Instance weight
 ·
 Wewill normalize the instance weight so that they can be summed
 up to 1 using the formula ��=��/���(�)Wi 
=Wi 
/sum(W)
 Step6– Repeat steps 2-5 for K iterations
 ·
 Wewill train K classifiers and will calculate model importance and
 update the instance weights using the above formula
 ·
 The final model M(X) will be an ensemble model which is obtained
 by combining these weak models weighted by their model weight
 """
