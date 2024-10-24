# Adaboost

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
AdaBoost is an ensemble learning method that builds a strong classifier by combining several weak classifiers. 
The core idea behind ensemble methods is to aggregate the predictions of multiple models to improve accuracy and reduce overfitting.

AdaBoost as an Ensemble Method:
1. Weak learners: It uses weak classifiers, often decision stumps (trees with one split), which perform only slightly better than random guessing.
2. Sequential training: AdaBoost trains these weak learners sequentially, where each learner focuses more on the samples that the previous learners misclassified.
3. Weighting: After each round of training, the algorithm adjusts the weights of the misclassified samples, giving more importance to difficult-to-classify examples.
4. Weighted majority vote: The final prediction is based on a weighted majority vote, where more accurate classifiers get higher weights.
By combining weak classifiers through boosting, AdaBoost creates a robust ensemble model that typically performs better than individual weak learners. 
It works well for binary classification but can be extended to multi-class tasks.
 """
