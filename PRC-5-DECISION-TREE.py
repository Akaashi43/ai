#AIM: Implement Decision-Tree learning algorithm.
#REQUIREMENTS:
"""
      1)---file1(balance-scale.data) & file2(balance-scale.names)
      2)---install these modules:
                  python -m pip install pandas==0.18
                  python -m pip install scipy
                  python -m pip install scikit-learn
                  python -m pip install numpy
"""

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#func importing dataset
def importdata():
      balance_data=pd.read_csv("balance-scale.data")

      #print the dataset shape
      print("Dataset Length : ",len(balance_data))
      
      #printing the dataset observations
      print("Dataset : ",balance_data.head())
      return balance_data

#func to split the dataset
def splitdataset(balance_data):
      #seperating the target variable
      X=balance_data.values[:,1:5]
      Y=balance_data.values[:,0]

      #splitting the dataset into train and test
      X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=100)
      return X,Y,X_train,X_test,y_train,y_test

#function to perform training with entropy
def train_using_entropy(X_train,X_test,y_train,y_test):
      #decision tree with entropy
      clf_entropy=DecisionTreeClassifier(criterion="entropy",random_state=100,max_depth=3,min_samples_leaf=5)

      #performing training
      clf_entropy.fit(X_train,y_train)
      return clf_entropy

def prediction(X_test,clf_object):
      y_pred=clf_object.predict(X_test)
      print("Predicted Values : ")
      print(y_pred)
      return y_pred

def cal_accuracy(y_test,y_pred):
      print("Accuracy : ",accuracy_score(y_test,y_pred)*100)

def main():
      data=importdata()
      X,Y,X_train,X_test,y_train,y_test=splitdataset(data)
      
      clf_entropy=train_using_entropy(X_train,X_test,y_train,y_test)

      print("Results using entropy : ")
      y_pred_entropy=prediction(X_test,clf_entropy)
      cal_accuracy(y_test,y_pred_entropy)

main()

'''
Decision trees are a popular and powerful tool used in various fields such as machine learning, data mining, and statistics. They provide a clear and intuitive way to make decisions based on data by modeling the relationships between different variables. This article is all about what decision trees are, how they work, their advantages and disadvantages, and their applications.
What is a Decision Tree?
A decision tree is a flowchart-like structure used to make decisions or predictions. It consists of nodes representing decisions or tests on attributes, branches representing the outcome of these decisions, and leaf nodes representing final outcomes or predictions. Each internal node corresponds to a test on an attribute, each branch corresponds to the result of the test, and each leaf node corresponds to a class label or a continuous value.
Structure of a Decision Tree
1.	Root Node: Represents the entire dataset and the initial decision to be made.
2.	Internal Nodes: Represent decisions or tests on attributes. Each internal node has one or more branches.
3.	Branches: Represent the outcome of a decision or test, leading to another node.
4.	Leaf Nodes: Represent the final decision or prediction. No further splits occur at these nodes.
How Decision Trees Work?
The process of creating a decision tree involves:
1.	Selecting the Best Attribute: Using a metric like Gini impurity, entropy, or information gain, the best attribute to split the data is selected.
2.	Splitting the Dataset: The dataset is split into subsets based on the selected attribute.
3.	Repeating the Process: The process is repeated recursively for each subset, creating a new internal node or leaf node until a stopping criterion is met (e.g., all instances in a node belong to the same class or a predefined depth is reached).

'''
