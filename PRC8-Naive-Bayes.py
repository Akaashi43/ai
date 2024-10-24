# Naive Bayes

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
import seaborn as sns
from joblib import dump

try:
    dataset = pd.read_csv('Social_Network_Ads.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: The dataset file was not found. Please check the file path.")
    raise

print(dataset.head())
print(dataset.info())

if dataset.isnull().sum().any():
    print("Missing values detected in the dataset.")

X = dataset.iloc[:, [2, 3]].values  
y = dataset.iloc[:, -1].values      

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_report(y_test, y_pred))

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Purchased', 'Purchased'], 
            yticklabels=['Not Purchased', 'Purchased'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

dump(classifier, 'naive_bayes_model.joblib')
print("Model saved as 'naive_bayes_model.joblib'.")

"""
# Naive Bayes learning algorithm

It is a classification technique based on Bayes' Theorem with an independence assumption among predictors.
In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.
The Naive Bayes classifier is a popular supervised machine learning algorithm used for classification tasks such as text classification. 
It belongs to the family of generative learning algorithms, which means that it models the distribution of inputs for a given class or category.
This approach is based on the assumption that the features of the input data are conditionally independent given the class, 
allowing the algorithm to make predictions quickly and accurately.

Example of Naive Bayes Algorithm
For example, if a fruit is red, round, and about 3 inches wide, we might call it an apple. Even if these things are related, 
each one helps us decide it’s probably an apple. That’s why it’s called ‘Naive.
An NB model is easy to build and particularly useful for very large data sets. Along with simplicity, 
Naive Bayes is known to outperform even highly sophisticated classification methods.
Bayes theorem provides a way of computing posterior probability P(c|x) from P(c), P(x) and P(x|c). 
"""
