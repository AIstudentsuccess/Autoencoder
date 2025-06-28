# Libraries 
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the main dataset
data = pd.read_csv("Cleaned_Sample_data.csv", low_memory=False)

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
---------------------------------

#------------------------------------------------------------------------------------------
# Random Forest Model On Input Data
#------------------------------------------------------------------------------------------

# Calculate class weights for the combined training and validation set
y_combined = pd.concat([y_train, y_val])
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_combined), y=y_combined)
weight_dict = {np.unique(y_combined)[i]: class_weights[i] for i in range(len(np.unique(y_combined)))}

# Initialize and train the Random Forest Classifier
rf = RandomForestClassifier(criterion='entropy', random_state=123, class_weight=weight_dict)
rf.fit(x_train_full, y_train)

# Predictions
y_val_pred = rf.predict(x_val_full)
y_test_pred = rf.predict(x_test_full)

# Compute confusion matrices
conf_matrix_val = confusion_matrix(y_val, y_val_pred)
conf_matrix_test = confusion_matrix(y_test, y_test_pred)

# Performance Metrics
metrics = {'Validation': {'Accuracy': accuracy_score(y_val, y_val_pred),
                          'F1 Score': f1_score(y_val, y_val_pred),
                          'Precision': precision_score(y_val, y_val_pred),
                          'Recall': recall_score(y_val, y_val_pred)},
           'Test': {'Accuracy': accuracy_score(y_test, y_test_pred),
                    'F1 Score': f1_score(y_test, y_test_pred),
                    'Precision': precision_score(y_test, y_test_pred),
                    'Recall': recall_score(y_test, y_test_pred)}}

metrics_df = pd.DataFrame(metrics)
print(metrics_df)

# Plotting ROC Curve
plt.figure(figsize=(4.5, 4.5))  
plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve on Input Data')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  
plt.legend(loc="lower right")
plt.show()

# Confusion Matrix Display for Validation and Test Sets
fig, ax = plt.subplots(1, 2, figsize=(10, 5))  

# Validation Set Confusion Matrix
disp_val = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_val, display_labels=rf.classes_)
disp_val.plot(cmap=plt.cm.Greys, ax=ax[0], values_format='.0f')  
ax[0].set_title('Validation Set on Input Data')

# Test Set Confusion Matrix
disp_test = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_test, display_labels=rf.classes_)
disp_test.plot(cmap=plt.cm.Greys, ax=ax[1], values_format='.0f')  
ax[1].set_title('Test Set on Input Data')
for disp in [disp_val, disp_test]:
    for text in disp.text_.ravel():
        bbox_patch = text.get_bbox_patch()
        if bbox_patch is not None:
            background_color = bbox_patch.get_facecolor()
            color_intensity = np.mean(background_color[:3])  
            text_color = 'white' if color_intensity < 0.5 else 'black'
            text.set_color(text_color)
        text.set_fontsize(12)  
plt.tight_layout()
plt.show()
