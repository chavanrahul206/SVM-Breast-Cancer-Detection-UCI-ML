import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
#Import Cancer data from the Sklearn library
# Dataset can also be found here (http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29)
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))
df_cancer.head()
df_cancer.shape
df_cancer.columns
# plot out just the first 5 variables (features)
sns.pairplot(df_cancer, vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness'] )
sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean perimeter','mean area','mean smoothness'] )
df_cancer['target'].value_counts()
sns.countplot(df_cancer['target'], label = "Count")
plt.figure(figsize=(20,12)) 
sns.heatmap(df_cancer.corr(), annot=True)
X = df_cancer.drop(['target'], axis = 1) # We drop our "target" feature and use all the remaining features in our dataframe to train the model.
X.head()
y = df_cancer['target']
y.head()

"""# Create the training and testing data

Now that we've assigned values to our "X" and "y", the next step is to import the python library that will help us to split our dataset into training and testing data.

- Training data = Is the subset of our data used to train our model.
- Testing data =  Is the subset of our data that the model hasn't seen before. This is used to test the performance of our model.
"""

from sklearn.model_selection import train_test_split

"""Let's split our data using 80% for training and the remaining 20% for testing."""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)

"""Let now check the size our training and testing data."""

print ('The size of our training "X" (input features) is', X_train.shape)
print ('\n')
print ('The size of our testing "X" (input features) is', X_test.shape)
print ('\n')
print ('The size of our training "y" (output feature) is', y_train.shape)
print ('\n')
print ('The size of our testing "y" (output features) is', y_test.shape)

"""# Import Support Vector Machine (SVM) Model"""

from sklearn.svm import SVC

svc_model = SVC()


svc_model.fit(X_train, y_train)


y_predict = svc_model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
cm = np.array(confusion_matrix(y_test, y_predict, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'],columns=['predicted_cancer','predicted_healthy'])
confusion
sns.heatmap(confusion, annot=True)
print(classification_report(y_test, y_predict))
X_train_min = X_train.min()
X_train_min

X_train_max = X_train.max()
X_train_max

X_train_range = (X_train_max- X_train_min)
X_train_range

X_train_scaled = (X_train - X_train_min)/(X_train_range)
X_train_scaled.head()

"""# Normalize Training Data"""

X_test_min = X_test.min()
X_test_range = (X_test - X_test_min).max()
X_test_scaled = (X_test - X_test_min)/X_test_range

svc_model = SVC()
svc_model.fit(X_train_scaled, y_train)

y_predict = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_predict)

"""# SVM with Normalized data"""

cm = np.array(confusion_matrix(y_test, y_predict, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'],
                         columns=['predicted_cancer','predicted_healthy'])
confusion

sns.heatmap(confusion,annot=True,fmt="d")

print(classification_report(y_test,y_predict))
