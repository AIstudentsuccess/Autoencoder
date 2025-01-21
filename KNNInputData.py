# Libraries 
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score, GridSearchCV, PredefinedSplit, KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GroupKFold

# Load the main dataset
data = pd.read_csv("Your File Path to modified_SSWD_POOL_V4.csv", low_memory=False)

#------------------------------------------------------------------------------------------
# Data preparation
#------------------------------------------------------------------------------------------

# Dropping the 'PROJECT_ID' and 'COHORTTRM_DESC' columns
data = data.drop(columns=['PROJECT_ID', 'COHORTTRM_DESC'])

# Separating features and target variable
x = data.drop(columns=['GRAD_YRS_ENCODED'])
y = data['GRAD_YRS_ENCODED']

# Splitting the data into training, validation, and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=123) # 0.25 x 0.8 = 0.2

# Explicitly creating copies of the slices to avoid SettingWithCopyWarning
x_train = x_train.copy()
x_val = x_val.copy()
x_test = x_test.copy()

# Columns to be standardized
columns_to_standardize = [
    'BESTMATH', 'BESTENGL', 'BESTCOMP', 'HS_ACAD_AVG', 'HS_OVERALL_AVG', 'AGE_MONTHS', 
    'EXP_FAM_CONTRIB', 'INCOME', 'YR1_FALL_SEM_GPA', 'YR1_SPRING_SEM_GPA', 'YR2_FALL_SEM_GPA', 
    'YR2_SPRING_SEM_GPA', 'YR1_FALL_LSU_GPA', 'YR1_SPRING_LSU_GPA', 'YR2_FALL_LSU_GPA', 
    'YR2_SPRING_LSU_GPA', 'YR1_FALL_CUM_GPA', 'YR1_SPRING_CUM_GPA', 'YR2_FALL_CUM_GPA', 
    'YR2_SPRING_CUM_GPA', 'YR1_FALL_CUM_HRS_CARR', 'YR1_SPRING_CUM_HRS_CARR', 'YR2_FALL_CUM_HRS_CARR', 
    'YR2_SPRING_CUM_HRS_CARR', 'YR1_FALL_CUM_HRS_EARN', 'YR1_SPRING_CUM_HRS_EARN', 'YR2_FALL_CUM_HRS_EARN', 
    'YR2_SPRING_CUM_HRS_EARN', 'AC_ACT1', 'AC_ACT2', 'AC_ACT3', 'AC_ACT4', 
    'HS_SCORE', 'Cumulative_Campus_Presence_Score', 'Longitude', 'Latitude'
]

# Standardizing the specified features
scaler = StandardScaler()
x_train[columns_to_standardize] = scaler.fit_transform(x_train[columns_to_standardize])
x_val[columns_to_standardize] = scaler.transform(x_val[columns_to_standardize])
x_test[columns_to_standardize] = scaler.transform(x_test[columns_to_standardize])

# Combine standardized with non-standardized for full datasets
x_train_full = pd.concat([x_train.drop(columns=columns_to_standardize), x_train[columns_to_standardize]], axis=1)
x_val_full = pd.concat([x_val.drop(columns=columns_to_standardize), x_val[columns_to_standardize]], axis=1)
x_test_full = pd.concat([x_test.drop(columns=columns_to_standardize), x_test[columns_to_standardize]], axis=1)

# Combine the training and validation sets for cross-validation
x_combined = pd.concat([x_train_full, x_val_full])
y_combined = pd.concat([y_train, y_val])

#------------------------------------------------------------------------------------------
# KNN model with random split
#------------------------------------------------------------------------------------------

# KNN Classifier with Grid Search and Cross-Validation
params = {'n_neighbors': list(range(1, 101))}  
knn = KNeighborsClassifier(metric='cosine')  

# Configure the KFold cross-validator
cv = KFold(n_splits=5, shuffle=True, random_state=123)  

grid_search = GridSearchCV(knn, params, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)  
grid_search.fit(x_combined, y_combined)

print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Evaluate on the test set
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(x_test_full)

# Reporting metrics
print(classification_report(y_test, y_pred))

#------------------------------------------------------------------------------------------
# KNN model with two-year gap and custom grouping
#------------------------------------------------------------------------------------------

# Note: To run the following code you need to restart the kernel. 

'''
In this current approach, we assume that training sets can start from any semester and training and testing sets are formed by selecting consecutive semesters,
and a two-year gap is maintained between the training and testing sets. Based on this assumption, eight possible configurations could maintain this gap.
Afterward, we performed the CV.
Training Groups            Testing Groups                
(1, 2, 3, 4)               (9, 10, 11, 12)               
(3, 4, 5, 6)               (11, 12, 13, 14)             
(5, 6, 7, 8)               (13, 14, 15, 16)            
(7, 8, 9, 10)              (15, 16, 17, 18)           
(9, 10, 11, 12)            (1, 2, 3, 4)                 
(11, 12, 13, 14)           (3, 4, 5, 6)                 
(13, 14, 15, 16)           (5, 6, 7, 8)                  
(15, 16, 17, 18)           (7, 8, 9, 10)                 
'''

# Map academic terms to sequential numbers for easier manipulation
term_to_group = {
    'Fall 2011': 1, 'Spring 2012': 2,
    'Fall 2012': 3, 'Spring 2013': 4,
    'Fall 2013': 5, 'Spring 2014': 6,
    'Fall 2014': 7, 'Spring 2015': 8,
    'Fall 2015': 9, 'Spring 2016': 10,
    'Fall 2016': 11, 'Spring 2017': 12,
    'Fall 2017': 13, 'Spring 2018': 14,
    'Fall 2018': 15, 'Spring 2019': 16,
    'Fall 2019': 17, 'Spring 2020': 18
}
data['group'] = data['COHORTTRM_DESC'].map(term_to_group)

# Prepare features and target
X = data.drop(['GRAD_YRS_ENCODED', 'PROJECT_ID', 'COHORTTRM_DESC'], axis=1)
y = data['GRAD_YRS_ENCODED']
groups = data['group']

# Columns to be standardized
columns_to_standardize = [
    'BESTMATH', 'BESTENGL', 'BESTCOMP', 'HS_ACAD_AVG', 'HS_OVERALL_AVG', 'AGE_MONTHS', 
    'EXP_FAM_CONTRIB', 'INCOME', 'YR1_FALL_SEM_GPA', 'YR1_SPRING_SEM_GPA', 'YR2_FALL_SEM_GPA', 
    'YR2_SPRING_SEM_GPA', 'YR1_FALL_LSU_GPA', 'YR1_SPRING_LSU_GPA', 'YR2_FALL_LSU_GPA', 
    'YR2_SPRING_LSU_GPA', 'YR1_FALL_CUM_GPA', 'YR1_SPRING_CUM_GPA', 'YR2_FALL_CUM_GPA', 
    'YR2_SPRING_CUM_GPA', 'YR1_FALL_CUM_HRS_CARR', 'YR1_SPRING_CUM_HRS_CARR', 'YR2_FALL_CUM_HRS_CARR', 
    'YR2_SPRING_CUM_HRS_CARR', 'YR1_FALL_CUM_HRS_EARN', 'YR1_SPRING_CUM_HRS_EARN', 'YR2_FALL_CUM_HRS_EARN', 
    'YR2_SPRING_CUM_HRS_EARN', 'AC_ACT1', 'AC_ACT2', 'AC_ACT3', 'AC_ACT4', 
    'HS_SCORE', 'Cumulative_Campus_Presence_Score', 'Longitude', 'Latitude'
]

# Standardize selected features
scaler = StandardScaler()
X[columns_to_standardize] = scaler.fit_transform(X[columns_to_standardize])

# KNN Classifier
knn = KNeighborsClassifier(metric='cosine', n_neighbors=24)

# Define the 8 configurations
configurations = [
    ((1, 2, 3, 4), (9, 10, 11, 12)), ((9, 10, 11, 12), (1, 2, 3, 4)),
    ((3, 4, 5, 6), (11, 12, 13, 14)), ((11, 12, 13, 14), (3, 4, 5, 6)),
    ((5, 6, 7, 8), (13, 14, 15, 16)), ((13, 14, 15, 16), (5, 6, 7, 8)),
    ((7, 8, 9, 10), (15, 16, 17, 18)), ((15, 16, 17, 18), (7, 8, 9, 10))
]

# Iterate over each configuration and perform training and testing
for train_groups, test_groups in configurations:
    train_indices = groups[groups.isin(train_groups)].index
    test_indices = groups[groups.isin(test_groups)].index

    X_train_fold, X_test_fold = X.iloc[train_indices], X.iloc[test_indices]
    y_train_fold, y_test_fold = y.iloc[train_indices], y.iloc[test_indices]

    # Train KNN on training folds
    knn.fit(X_train_fold, y_train_fold)

    # Validate KNN on the test fold
    y_pred = knn.predict(X_test_fold)
    print(f"Training Groups: {train_groups}, Testing Groups: {test_groups}")
    print(classification_report(y_test_fold, y_pred))

# Final evaluation on the test set
# For the final evaluation, we can use one of the configurations or define a new one based on the data characteristics.
# Here, we reuse one of the configurations for simplicity:
train_groups = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
test_groups = (15, 16, 17, 18)
train_indices = groups[groups.isin(train_groups)].index
test_indices = groups[groups.isin(test_groups)].index

X_train_final, X_test_final = X.iloc[train_indices], X.iloc[test_indices]
y_train_final, y_test_final = y.iloc[train_indices], y.iloc[test_indices]

# Train final model
knn.fit(X_train_final, y_train_final)

# Evaluate on final test set
y_pred_final = knn.predict(X_test_final)
print("Final Test Set Results:")
print(classification_report(y_test_final, y_pred_final))
