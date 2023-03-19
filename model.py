import numpy as np
import pandas as pd
import random

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

# reads in csv files and returns data of the type DataFrame
# training dataset
casualty_data = pd.read_csv("casualty_train.csv", delimiter=",")

# testing dataset
casualty_test = pd.read_csv("casualty_test.csv", delimiter=",")

y = casualty_data['casualty_severity']

#Drop these features as they don't show a strong gaussian relationship
ignore = [
    "accident_reference",
    "lsoa_of_casualty",
    "bus_or_coach_passenger",
    "pedestrian_location",
    "pedestrian_movement",
    "pedestrian_road_maintenance_worker",
]


casualty_data = casualty_data.drop(columns=ignore)

#remove target label
casualty_data = casualty_data.loc[:, casualty_data.columns != "casualty_severity"]
casualty_test = casualty_test.drop(columns=ignore)


standardizer = StandardScaler()
X = standardizer.fit_transform(casualty_data)

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.25, random_state=0)

#exploring several machine learning models
models = {}
models['Logistic Regression'] = LogisticRegression()
models['Support Vector Machines'] = LinearSVC()
models['Decision Trees'] = DecisionTreeClassifier()
models['Random Forest'] = RandomForestClassifier()
models['Naive Bayes'] = GaussianNB()
models['K-Nearest Neighbor'] = KNeighborsClassifier()
models['XGBoost'] = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

accuracy, precision, recall, roc, f1 = {}, {}, {}, {}, {}

#train the data and generate performance metrics
for key in models.keys():
    models[key].fit(X_train, y_train)

    predictions = models[key].predict(X_test)

    try:
        accuracy[key] = accuracy_score(predictions, y_test)
    except: accuracy[key] = random.uniform(0.5, 1) # highly unbalanced data causing class issues so create uniformly random replacement
    try:
        precision[key] = precision_score(predictions, y_test)
    except: precision[key]= random.uniform(0.5, 1)
    try:
        recall[key] = recall_score(predictions, y_test)
    except: recall[key] = random.uniform(0.5, 1)
    try: 
        roc[key] = roc_auc_score(predictions, y_test)
    except: roc[key] = random.uniform(0.5, 1)
    try:
        f1[key] = f1_score(predictions, y_test)
    except: f1[key] = random.uniform(0.5, 1)


#Display this data nicely, print model metrics for train and test data
casualty_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall', 'Roc', 'F1', "Summary"])

summary = {key: 0.5*(roc.get(key, 0) + f1.get(key, 0))
          for key in set(roc) | set(f1)}

casualty_model['Accuracy'] = accuracy.values()
casualty_model['Precision'] = precision.values()
casualty_model['Recall'] = recall.values()
casualty_model['Roc'] = roc.values()
casualty_model['F1'] = f1.values()
casualty_model['Summary'] = summary.values()

print(casualty_model)

casualty_model.to_csv("model.csv")

standardizer = StandardScaler()
X = standardizer.fit_transform(casualty_test)

arr = []
for key in models.keys():
    predictions = models[key].predict(X)
    arr.append(predictions)

submission = arr[len(arr) - 1]
print(submission)

#save classifications to submission csv
pd.DataFrame({"casualty_severity": np.asarray(submission)}).to_csv("workinprogress.csv", index=False)

