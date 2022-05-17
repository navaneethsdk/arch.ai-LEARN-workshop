import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle

data = pd.read_csv('Data/Gesture_grouped_Data.csv')
# data.head()

# Data processing
X = data.drop(columns=['label'])
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=109)

# SVM model for classifcation
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

filename = './Models/model.sav'
pickle.dump(clf, open(filename, 'wb'))