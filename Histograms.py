# Libraries 
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.manifold import TSNE

# -----------------------------------------------------------------------------
# Examine the dataset
# -----------------------------------------------------------------------------

# Load the main dataset
data = pd.read_csv("cleaned_sample_data.csv", low_memory=False)

# Print raw data information
num_rows, num_columns = data.shape
print(f"The dataset has {num_rows} rows and {num_columns} columns.")

# Print only the column names
for col in data.columns:
    print(col)

# print features and number of missing values
for col in data.columns:
    missing_count = data[col].isnull().sum()
    if missing_count > 0:
        print(f"{col}: {missing_count} missing values")
    else:
        print(f"{col}: No missing values")

# List of columns for which to check data types
columns_to_check = [
    'BESTMATH', 'BESTENGL', 'BESTCOMP', 'HS_ACAD_AVG', 'HS_OVERALL_AVG', 'AGE_MONTHS',
    'EXP_FAM_CONTRIB', 'INCOME', 'PELL', 'TOPS', 'GREEK_ENTRY', 'YR1_FALL_SEM_GPA',
    'YR1_SPRING_SEM_GPA', 'YR2_FALL_SEM_GPA', 'YR2_SPRING_SEM_GPA', 'YR1_FALL_LSU_GPA',
    'YR1_SPRING_LSU_GPA', 'YR2_FALL_LSU_GPA', 'YR2_SPRING_LSU_GPA', 'YR1_FALL_CUM_GPA',
    'YR1_SPRING_CUM_GPA', 'YR2_FALL_CUM_GPA', 'YR2_SPRING_CUM_GPA', 'YR1_FALL_CUM_HRS_CARR',
    'YR1_SPRING_CUM_HRS_CARR', 'YR2_FALL_CUM_HRS_CARR', 'YR2_SPRING_CUM_HRS_CARR',
    'YR1_FALL_CUM_HRS_EARN', 'YR1_SPRING_CUM_HRS_EARN', 'YR2_FALL_CUM_HRS_EARN',
    'YR2_SPRING_CUM_HRS_EARN', 'AC_ACT1', 'AC_ACT2', 'AC_ACT3', 'AC_ACT4', 'Latitude',
    'Longitude', 'CH_ENRLTYPE_encoded', 'CHRTTYPE_encoded', 'CRSLOAD_encoded',
    'GENDER_encoded', 'FIRST_GEN_FLAG_encoded',
    'COUNTRY_DESC_encoded', 'GRAD_YRS_ENCODED', 'HS_SCORE', 'Cumulative_Campus_Presence_Score',
    'Curriculum_ACCT', 'College_ADSN', 'Curric_College_ADSN',
    'HS_Type_Charter', 'HS_Type_Home School', 'HS_Type_Independent', 'HS_Type_Public', 'HS_Type_Religious', 'HS_Type_nan'
]

# Print data types of specified columns
for col in columns_to_check:
    if col in data.columns:
        print(f"Data type of {col}: {data[col].dtype}")
    else:
        print(f"{col} is not a column in the DataFrame")

# List of columns for which to find the range
columns_to_check = [
    'BESTMATH', 'BESTENGL', 'BESTCOMP', 'HS_ACAD_AVG', 'HS_OVERALL_AVG', 'AGE_MONTHS',
    'EXP_FAM_CONTRIB', 'INCOME', 'PELL', 'TOPS', 'GREEK_ENTRY', 'YR1_FALL_SEM_GPA',
    'YR1_SPRING_SEM_GPA', 'YR2_FALL_SEM_GPA', 'YR2_SPRING_SEM_GPA', 'YR1_FALL_LSU_GPA',
    'YR1_SPRING_LSU_GPA', 'YR2_FALL_LSU_GPA', 'YR2_SPRING_LSU_GPA', 'YR1_FALL_CUM_GPA',
    'YR1_SPRING_CUM_GPA', 'YR2_FALL_CUM_GPA', 'YR2_SPRING_CUM_GPA', 'YR1_FALL_CUM_HRS_CARR',
    'YR1_SPRING_CUM_HRS_CARR', 'YR2_FALL_CUM_HRS_CARR', 'YR2_SPRING_CUM_HRS_CARR',
    'YR1_FALL_CUM_HRS_EARN', 'YR1_SPRING_CUM_HRS_EARN', 'YR2_FALL_CUM_HRS_EARN',
    'YR2_SPRING_CUM_HRS_EARN', 'AC_ACT1', 'AC_ACT2', 'AC_ACT3', 'AC_ACT4', 'Latitude',
    'Longitude', 'CH_ENRLTYPE_encoded', 'CHRTTYPE_encoded', 'CRSLOAD_encoded',
    'GENDER_encoded', 'FIRST_GEN_FLAG_encoded',
    'COUNTRY_DESC_encoded', 'GRAD_YRS_ENCODED', 'HS_SCORE', 'Cumulative_Campus_Presence_Score',
    'Curriculum_ACCT', 'College_ADSN', 'Curric_College_ADSN',
    'HS_Type_Charter', 'HS_Type_Independent', 'HS_Type_Public', 'HS_Type_Religious', 'HS_Type_nan'
]

# Note: 'HS_Type_Home School' is not a column in the cleaned sample data; however, it exists in the entire dataset. 

# Calculate and print the range for each column
for col in columns_to_check:
    min_value = data[col].min()
    max_value = data[col].max()
    range_of_values = max_value - min_value
    print(f"Range of {col}: {range_of_values} (Min: {min_value}, Max: {max_value})")

# -----------------------------------------------------------------------------
# Plot histograms before standardization
# -----------------------------------------------------------------------------

# List of columns for which to plot histograms
columns_to_plot = [
    'BESTMATH', 'BESTENGL', 'BESTCOMP', 'HS_ACAD_AVG', 'HS_OVERALL_AVG', 'AGE_MONTHS',
    'EXP_FAM_CONTRIB', 'INCOME',
    'YR1_FALL_SEM_GPA', 'YR1_SPRING_SEM_GPA', 'YR2_FALL_SEM_GPA', 'YR2_SPRING_SEM_GPA',
    'YR1_FALL_LSU_GPA', 'YR1_SPRING_LSU_GPA', 'YR2_FALL_LSU_GPA', 'YR2_SPRING_LSU_GPA',
    'YR1_FALL_CUM_GPA', 'YR1_SPRING_CUM_GPA', 'YR2_FALL_CUM_GPA', 'YR2_SPRING_CUM_GPA',
    'YR1_FALL_CUM_HRS_CARR', 'YR1_SPRING_CUM_HRS_CARR', 'YR2_FALL_CUM_HRS_CARR', 'YR2_SPRING_CUM_HRS_CARR',
    'YR1_FALL_CUM_HRS_EARN', 'YR1_SPRING_CUM_HRS_EARN', 'YR2_FALL_CUM_HRS_EARN', 'YR2_SPRING_CUM_HRS_EARN'
]
'''
# Create a histogram for each column
for col in columns_to_plot:
    plt.figure(figsize=(10, 4))
    plt.hist(data[col].dropna(), bins=30, edgecolor='k', alpha=0.7)
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()
'''
# Create a histogram for each column
for col in columns_to_plot:
    plt.figure(figsize=(10, 4))

    # Use a logarithmic scale for 'EXP_FAM_CONTRIB' and 'INCOME'
    if col in ['EXP_FAM_CONTRIB', 'INCOME']:
        data_col = data[col].dropna()
        # Replace zero values with a small number for logarithmic scale
        data_col = data_col.replace(0, np.nextafter(0, 1))
        plt.hist(data_col, bins=30, edgecolor='k', alpha=0.7, log=True, color='silver')
        plt.xlabel(f'{col} (log scale)', fontsize=18)
    else:
        plt.hist(data[col].dropna(), bins=30, edgecolor='k', alpha=0.7, color='silver')
        plt.xlabel(col, fontsize=18)

    #plt.title(f'Histogram of {col}')
    plt.ylabel('Frequency', fontsize=18)
    plt.show()

 # List of categorical columns for which to plot pie charts
categorical_columns = [
    'HS_SCORE', 'Cumulative_Campus_Presence_Score', 'AC_ACT1', 'AC_ACT2', 'AC_ACT3', 'AC_ACT4', 
    'PELL', 'TOPS', 'GREEK_ENTRY', 'CH_ENRLTYPE_encoded', 'CHRTTYPE_encoded', 'CRSLOAD_encoded', 
    'GENDER_encoded', 'FIRST_GEN_FLAG_encoded', 'COUNTRY_DESC_encoded', 
    'GRAD_YRS_ENCODED',
    'HS_Type_Charter', 'HS_Type_Independent', 'HS_Type_Public', 'HS_Type_Religious', 'HS_Type_nan'
]

# Note: 'HS_Type_Home School' is not a column in the cleaned sample data; however, it exists in the entire dataset. 

# Create a pie chart for each categorical column
for col in categorical_columns:
    plt.figure(figsize=(8, 8))
    data[col].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title(f'Pie Chart of {col}')
    plt.ylabel('')
    plt.show()

# -----------------------------------------------------------------------------
# Plot histograms after applying standardization                                                                       
# -----------------------------------------------------------------------------

# Dropping the 'PROJECT_ID' and 'COHORTTRM_DESC' columns
data = data.drop(columns=['PROJECT_ID', 'COHORTTRM_DESC'])

# Separating features and target variable
x = data.drop(columns=['GRAD_YRS_ENCODED'])
y = data['GRAD_YRS_ENCODED']

# Splitting the data into training, validation, and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2

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

# Standardizing the specified features in the training set
scaler = StandardScaler()
x_train[columns_to_standardize] = scaler.fit_transform(x_train[columns_to_standardize])
x_val[columns_to_standardize] = scaler.transform(x_val[columns_to_standardize])
x_test[columns_to_standardize] = scaler.transform(x_test[columns_to_standardize])

# Convert the entire DataFrame to tensors 
x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
x_val_tensor = torch.tensor(x_val.values, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# List of numerical columns for which to plot histograms after standardization
numerical_columns = [
    'BESTMATH', 'BESTENGL', 'BESTCOMP', 'HS_ACAD_AVG', 'HS_OVERALL_AVG', 'AGE_MONTHS',
    'EXP_FAM_CONTRIB', 'INCOME',
    'YR1_FALL_SEM_GPA', 'YR1_SPRING_SEM_GPA', 'YR2_FALL_SEM_GPA', 'YR2_SPRING_SEM_GPA',
    'YR1_FALL_LSU_GPA', 'YR1_SPRING_LSU_GPA', 'YR2_FALL_LSU_GPA', 'YR2_SPRING_LSU_GPA',
    'YR1_FALL_CUM_GPA', 'YR1_SPRING_CUM_GPA', 'YR2_FALL_CUM_GPA', 'YR2_SPRING_CUM_GPA',
    'YR1_FALL_CUM_HRS_CARR', 'YR1_SPRING_CUM_HRS_CARR', 'YR2_FALL_CUM_HRS_CARR', 'YR2_SPRING_CUM_HRS_CARR',
    'YR1_FALL_CUM_HRS_EARN', 'YR1_SPRING_CUM_HRS_EARN', 'YR2_FALL_CUM_HRS_EARN', 'YR2_SPRING_CUM_HRS_EARN'
]

# Create a histogram for each column after standardization
for col in numerical_columns:
    plt.figure(figsize=(10, 4))
    plt.hist(x_train[col].dropna(), bins=30, edgecolor='k', alpha=0.7)
    plt.xlabel(f'{col} (Standardized)')
    plt.title(f'Histogram of {col} (Standardized)')
    plt.ylabel('Frequency')
    plt.show()
    
# List of categorical columns for which to plot pie charts after standardization
categorical_columns = [
    'HS_SCORE', 'Cumulative_Campus_Presence_Score', 'AC_ACT1', 'AC_ACT2', 'AC_ACT3', 'AC_ACT4'
]

# Create a pie chart for each categorical column after standardization
for col in categorical_columns:
    if col in x_train.columns:  
        plt.figure(figsize=(8, 8))
        x_train[col].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title(f'Pie Chart of {col} (After Normalization)')
        plt.ylabel('') 
        plt.show()

# -----------------------------------------------------------------------------
# Plot histograms after applying standardization on the entire dataset                                                                   
# -----------------------------------------------------------------------------

# Note: To see the effect of standardization on the entire dataset not just to the X_train subset you need to restart your kernel, run the libraries, and load the main dataset again. 

# Dropping the 'PROJECT_ID' and 'COHORTTRM_DESC' columns
data = data.drop(columns=['PROJECT_ID', 'COHORTTRM_DESC'])

# Separating features and target variable
x = data.drop(columns=['GRAD_YRS_ENCODED'])
y = data['GRAD_YRS_ENCODED']

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

scaler = StandardScaler()
x_standardized = x.copy()
x_standardized[columns_to_standardize] = scaler.fit_transform(x[columns_to_standardize])

# List of numerical columns for which to plot histograms after standardization
numerical_columns = [
    'BESTMATH', 'BESTENGL', 'BESTCOMP', 'HS_ACAD_AVG', 'HS_OVERALL_AVG', 'AGE_MONTHS',
    'EXP_FAM_CONTRIB', 'INCOME', 
    'YR1_FALL_SEM_GPA', 'YR1_SPRING_SEM_GPA', 'YR2_FALL_SEM_GPA', 'YR2_SPRING_SEM_GPA',
    'YR1_FALL_LSU_GPA', 'YR1_SPRING_LSU_GPA', 'YR2_FALL_LSU_GPA', 'YR2_SPRING_LSU_GPA',
    'YR1_FALL_CUM_GPA', 'YR1_SPRING_CUM_GPA', 'YR2_FALL_CUM_GPA', 'YR2_SPRING_CUM_GPA',
    'YR1_FALL_CUM_HRS_CARR', 'YR1_SPRING_CUM_HRS_CARR', 'YR2_FALL_CUM_HRS_CARR', 'YR2_SPRING_CUM_HRS_CARR',
    'YR1_FALL_CUM_HRS_EARN', 'YR1_SPRING_CUM_HRS_EARN', 'YR2_FALL_CUM_HRS_EARN', 'YR2_SPRING_CUM_HRS_EARN'
]

# Create a histogram for each column in the standardized dataset
for col in numerical_columns:
    plt.figure(figsize=(10, 4))
    plt.hist(x_standardized[col].dropna(), bins=30, edgecolor='k', alpha=0.7, color='silver')
    #plt.title(f'Histogram of {col} (Standardized)')
    plt.xlabel(f'{col} (Standardized)', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.show()  

# List of categorical columns for which to plot pie charts after standardization
categorical_columns = [
    'HS_SCORE', 'Cumulative_Campus_Presence_Score', 'AC_ACT1', 'AC_ACT2', 'AC_ACT3', 'AC_ACT4'
]

# Create a pie chart for each categorical column after standardization
for col in categorical_columns:
    if col in x_standardized.columns:  # Check if the column is in the standardized DataFrame
        plt.figure(figsize=(8, 8))
        x_standardized[col].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title(f'Pie Chart of {col} (Standardized)')
        plt.ylabel('')  # Hide the y-label
        plt.show()  
