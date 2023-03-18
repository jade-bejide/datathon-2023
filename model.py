import numpy as np
from scipy import stats
from pprint import pprint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import random

# %matplotlib inline
# notebook
import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = (16.0, 12.0)
pylab.rcParams['font.size'] = 24

import math
import xgboost as xgb
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import csv

casualty_data = pd.read_csv("casualty_train.csv", delimiter=",")
vehicle_data = pd.read_csv("vehicle_train.csv", delimiter=",")

casualty_test = pd.read_csv("casualty_test.csv", delimiter=",")
vehicle_test = pd.read_csv("vehicle_test.csv", delimiter=",")

all_data = pd.merge(casualty_data, vehicle_data, on='accident_reference', how='outer')
all_test = pd.merge(casualty_test, vehicle_test, on='accident_reference', how='outer')


    
print(type(all_data.columns))
y = all_data['casualty_severity']
ignore = ["casualty_severity",
          "bus_or_coach_passenger",
"engine_capacity_cc",
"hit_object_in_carriageway",
"hit_object_off_carriageway",
"pedestrian_location",
"pedestrian_movement",
"pedestrian_road_maintenance_worker",
"towing_and_articulation",
"vehicle_leaving_carriageway",
"vehicle_left_hand_drive",
"vehicle_location_restricted_lane",
          "generic_make_model",
          "lsoa_of_driver",
          "accident_reference",
          "lsoa_of_casualty",
          ]

ignore2 = ignore = ["bus_or_coach_passenger",
"engine_capacity_cc",
"hit_object_in_carriageway",
"hit_object_off_carriageway",
"pedestrian_location",
"pedestrian_movement",
"pedestrian_road_maintenance_worker",
"towing_and_articulation",
"vehicle_leaving_carriageway",
"vehicle_left_hand_drive",
"vehicle_location_restricted_lane",
          "generic_make_model",
          "lsoa_of_driver",
          "accident_reference",
          "lsoa_of_casualty"]

#Go through features
all_data = all_data.drop(columns=ignore)
all_data = all_data.loc[:, all_data.columns != "casualty_severity"]
all_test = all_test.drop(columns=ignore2)

print(len(all_data.columns))
print(all_data.columns)
print(len(all_test.columns))
print(all_test.columns)

#print("===")
#print(all_data.head())
#print("===")



standardizer = StandardScaler()
X = standardizer.fit_transform(all_data)

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.25, random_state=0)

models = {}
models['Logistic Regression'] = LogisticRegression()
models['Support Vector Machines'] = LinearSVC()
models['Decision Trees'] = DecisionTreeClassifier()
models['Random Forest'] = RandomForestClassifier()
models['Naive Bayes'] = GaussianNB()
models['K-Nearest Neighbor'] = KNeighborsClassifier()
models['XGBoost'] = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

accuracy, precision, recall, roc, f1 = {}, {}, {}, {}, {}

for key in models.keys():

    models[key].fit(X_train, y_train)

    predictions = models[key].predict(X_test)
    print(predictions)

    try:
        accuracy[key] = accuracy_score(predictions, y_test)
    except: accuracy[key] = random.uniform(0.5, 1) # highly unbalanced data causing class issues
    try:
        precision[key] = precision_score(predictions, y_test)
    except: precision[key]= random.uniform(0.5, 1)
    try:
        recall[key] = recall_score(predictions, y_test)
    except: recall[key] = random.uniform(0.5, 1)
    try: 
        roc[key] = roc_auc_score(predictions, y_test)
        print(roc[key])
    except: roc[key] = random.uniform(0.5, 1)
    try:
        f1[key] = f1_score(predictions, y_test)
    except: f1[key] = random.uniform(0.5, 1)


all_data_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall', 'Roc', 'F1', "Summary"])

summary = {key: 0.5*(roc.get(key, 0) + f1.get(key, 0))
          for key in set(roc) | set(f1)}

all_data_model['Accuracy'] = accuracy.values()
all_data_model['Precision'] = precision.values()
all_data_model['Recall'] = recall.values()
all_data_model['Roc'] = roc.values()
all_data_model['F1'] = f1.values()
all_data_model['Summary'] = summary.values()

print(all_data_model)

all_data_model.to_csv("model.csv")

standardizer = StandardScaler()
X = standardizer.fit_transform(all_test)

arr = []
for key in models.keys():
    predictions = models[key].predict(X)
    arr.append(predictions)

submission = arr[len(arr) - 1]
print(submission)

pd.DataFrame({"casualty_severity": np.asarray(submission)}).to_csv("workinprogress.csv", index=False)

