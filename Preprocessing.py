# Libraries 
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import folium
from folium.plugins import HeatMap

# -----------------------------------------------------------------------------
# Data preprocessing
# -----------------------------------------------------------------------------

# Load the main dataset
data = pd.read_csv("C:\\Users\\e_sho\\Desktop\\SSWD_POOL_V1.csv", low_memory=False)

# Load the zip code coordinates dataset
zip_coordinates = pd.read_csv("C:\\Users\\e_sho\\Desktop\\zip_code_coordinates.csv", low_memory=False)

# Load the country coordinates dataset
country_coordinates = pd.read_csv("C:\\Users\\e_sho\\Desktop\\country_and_state_coordinates.csv", low_memory=False)

# Print raw data information
num_rows, num_columns = data.shape
print(f"The dataset has {num_rows} rows and {num_columns} columns.")

# examine the dataset
print(data.info())

# print features and number of missing values
for col in data.columns:
    missing_count = data[col].isnull().sum()
    if missing_count > 0:
        print(f"{col}: {missing_count} missing values")
    else:
        print(f"{col}: No missing values")
        
# Filter out rows where ATHLETE is 1 or VETERAN_STATUS is 'Y'
data = data[(data['ATHLETE'] != 1) & (data['VETERAN_STATUS'] != 'Y')]

# Data Encoding and Transformations
data['CH_ENRLTYPE_encoded'] = data['CH_ENRLTYPE'].map({'Campus': 1, 'Online': 0})
data['CHRTTYPE_encoded'] = data['CHRTTYPE'].map({'F': 1, 'T': 0})
data['CRSLOAD_encoded'] = data['CRSLOAD'].map({'Full-time': 1, 'Part-time': 0})
data['PVT_SCHOOL_FLAG_encoded'] = data['PVT_SCHOOL_FLAG'].map({'Y': 2, 'N': 0, 'U': 1}).fillna(1)
data['GENDER_encoded'] = data['GENDER'].map({'M': 1, 'F': 0})
data['FIRST_GEN_FLAG_encoded'] = data['FIRST_GEN_FLAG'].map({'Y': 2, 'N': 0, 'U': 1}).fillna(1)
data['COUNTRY_DESC_encoded'] = data['COUNTRY_DESC'].apply(lambda x: 0 if x == 'United States' else 1)
data['GREEK_ENTRY'].fillna(0, inplace=True)

# One-hot encode 'DEGR_PROGRAM'
degr_program_dummies = pd.get_dummies(data['DEGR_PROGRAM'], prefix='Program')
data = pd.concat([data, degr_program_dummies], axis=1)

# One-hot encode 'DEGR_COLLEGE'
degr_college_dummies = pd.get_dummies(data['DEGR_COLLEGE'], prefix='College')
data = pd.concat([data, degr_college_dummies], axis=1)

# One-hot encode 'CH_CURRICULUM_1'
curriculum_dummies = pd.get_dummies(data['CH_CURRICULUM_1'], prefix='Curriculum')
data = pd.concat([data, curriculum_dummies], axis=1)

# One-hot encode 'CH_COLLEGE_1'
college_dummies = pd.get_dummies(data['CH_COLLEGE_1'], prefix='College')
data = pd.concat([data, college_dummies], axis=1)

# One-hot encode 'CH_CURRIC_COLLEGE'
curric_college_dummies = pd.get_dummies(data['CH_CURRIC_COLLEGE'], prefix='Curric_College')
data = pd.concat([data, curric_college_dummies], axis=1)

# -----------------------------------------------------------------------------
# High School Score Calculation 
# -----------------------------------------------------------------------------

# Define the scores for each category
scores = {
    'HS_TOP_10': 5,
    'HS_TOP_25': 12.5,
    'HS_TOP_50': 25,
    'HS_BOTTOM_25': 87.5,
    'HS_BOTTOM_50': 75
}

# Calculate the arithmetic average of the scores
average_score = sum(scores.values()) / len(scores)

# Function to calculate the score for each student
def calculate_hs_score(row):
    for category, score in scores.items():
        if row[category] == 1:
            return score
    return average_score  # Return average score for missing values

# Apply the function to each row
data['HS_SCORE'] = data.apply(calculate_hs_score, axis=1)

# Now 'data' has a new column 'HS_SCORE' with the calculated scores

# --------------------------------------------------------------------------------------
# Replacing BESTMATH, BESTENGL, BESTCOMP, HS_ACAD_AVG, HS_OVERALL_AVG missing values
# --------------------------------------------------------------------------------------

# Columns to process
columns_to_process = ['BESTMATH', 'BESTENGL', 'BESTCOMP', 'HS_ACAD_AVG', 'HS_OVERALL_AVG']

# Calculate and fill missing values for each column
for column in columns_to_process:
    # Calculate median for each HS_SCORE
    median_scores = data.groupby('HS_SCORE')[column].median()

    # Find the minimum non-zero value in the column
    min_non_zero = data[data[column] > 0][column].min()

    # Function to get median score for a given HS_SCORE
    def get_median_score(hs_score):
        return median_scores.get(hs_score)

    # Fill missing values and replace zeros
    data[column] = data.apply(
        lambda row: get_median_score(row['HS_SCORE']) if pd.isnull(row[column]) else (min_non_zero - 1 if row[column] == 0 else row[column]),
        axis=1
    )

# -----------------------------------------------------------------------------
# Campus Presence Score Calculation
# -----------------------------------------------------------------------------

# List of columns to consider
campus_presence_columns = [
    'YR1_FALL_ON_CAMPUS', 'YR1_SPRING_ON_CAMPUS',
    'YR2_FALL_ON_CAMPUS', 'YR2_SPRING_ON_CAMPUS',
    'YR3_FALL_ON_CAMPUS', 'YR3_SPRING_ON_CAMPUS',
    'YR4_FALL_ON_CAMPUS', 'YR4_SPRING_ON_CAMPUS',
    'YR5_FALL_ON_CAMPUS', 'YR5_SPRING_ON_CAMPUS',
    'YR6_FALL_ON_CAMPUS', 'YR6_SPRING_ON_CAMPUS',
    'YR7_FALL_ON_CAMPUS', 'YR7_SPRING_ON_CAMPUS',
    'YR8_FALL_ON_CAMPUS', 'YR8_SPRING_ON_CAMPUS'
]

# Convert 'Y' to 1, 'N' to 0, and NaN to 0 ( treating missing as absence)
for col in campus_presence_columns:
    data[col] = data[col].map({'Y': 1, 'N': 0}).fillna(0)

# Calculate the Cumulative Campus Presence Score
data['Cumulative_Campus_Presence_Score'] = data[campus_presence_columns].sum(axis=1)

# Now 'data' has a new column 'Cumulative_Campus_Presence_Score' with the calculated scores

# -----------------------------------------------------------------------------
# Handling the first year missing values for SEM_GPAs
# -----------------------------------------------------------------------------

def calculate_median_gpas(data, year):
    """
    Calculate the median GPA for each student based on the year of study.

    :param data: DataFrame containing GPA columns.
    :param year: Year of study (1 for first year, 2 for second year, etc.)
    :return: Series with median GPA for each student.
    """
    semester_columns = [f'YR{yr}_{"FALL" if sem == 1 else "SPRING"}_SEM_GPA' for yr in range(1, year+1) for sem in range(1, 3)]
    return data[semester_columns].median(axis=1, skipna=True)

def fill_missing_gpas(data, year):
    """
    Fill missing GPAs with the median GPA of the available semesters based on the year of study.

    :param data: DataFrame containing GPA columns.
    :param year: Year of study (1 for first year, 2 for second year, etc.)
    """
    median_gpas = calculate_median_gpas(data, year)
    semester_columns = [f'YR{yr}_{"FALL" if sem == 1 else "SPRING"}_SEM_GPA' for yr in range(1, year+1) for sem in range(1, 3)]

    for col in semester_columns:
        data[col] = data[col].fillna(median_gpas)

# Example usage
year_of_study = 1  
fill_missing_gpas(data, year_of_study)

# -----------------------------------------------------------------------------
# Handling the second year missing values for SEM_GPAs
# -----------------------------------------------------------------------------
def calculate_median_gpas(data, year):
    """
    Calculate the median GPA for each student based on the year of study.

    :param data: DataFrame containing GPA columns.
    :param year: Year of study (1 for first year, 2 for second year, etc.)
    :return: Series with median GPA for each student.
    """
    semester_columns = [f'YR{yr}_{"FALL" if sem == 1 else "SPRING"}_SEM_GPA' for yr in range(1, year+1) for sem in range(1, 3)]
    return data[semester_columns].median(axis=1, skipna=True)

def fill_missing_gpas(data, year):
    """
    Fill missing GPAs with the median GPA of the available semesters based on the year of study.

    :param data: DataFrame containing GPA columns.
    :param year: Year of study (1 for first year, 2 for second year, etc.)
    """
    median_gpas = calculate_median_gpas(data, year)
    semester_columns = [f'YR{yr}_{"FALL" if sem == 1 else "SPRING"}_SEM_GPA' for yr in range(1, year+1) for sem in range(1, 3)]

    for col in semester_columns:
        data[col] = data[col].fillna(median_gpas)

# Example usage
year_of_study = 2  
fill_missing_gpas(data, year_of_study)

# -----------------------------------------------------------------------------
# Handling the first year missing values for LSU_GPAs
# -----------------------------------------------------------------------------

def calculate_median_gpas(data, year):
    """
    Calculate the median LSU GPA for each student based on the year of study.

    :param data: DataFrame containing LSU GPA columns.
    :param year: Year of study (1 for first year, 2 for second year, etc.)
    :return: Series with median LSU GPA for each student.
    """
    semester_columns = [f'YR{yr}_{"FALL" if sem == 1 else "SPRING"}_LSU_GPA' for yr in range(1, year+1) for sem in range(1, 3)]
    return data[semester_columns].median(axis=1, skipna=True)

def fill_missing_gpas(data, year):
    """
    Fill missing LSU GPAs with the median LSU GPA of the available semesters based on the year of study.

    :param data: DataFrame containing LSU GPA columns.
    :param year: Year of study (1 for first year, 2 for second year, etc.)
    """
    median_gpas = calculate_median_gpas(data, year)
    semester_columns = [f'YR{yr}_{"FALL" if sem == 1 else "SPRING"}_LSU_GPA' for yr in range(1, year+1) for sem in range(1, 3)]

    for col in semester_columns:
        data[col] = data[col].fillna(median_gpas)

# Example usage
year_of_study = 1  
fill_missing_gpas(data, year_of_study)

# -----------------------------------------------------------------------------
# Handling the second year missing values for LSU_GPAs
# -----------------------------------------------------------------------------

def calculate_median_gpas(data, year):
    """
    Calculate the median LSU GPA for each student based on the year of study.

    :param data: DataFrame containing LSU GPA columns.
    :param year: Year of study (1 for first year, 2 for second year, etc.)
    :return: Series with median LSU GPA for each student.
    """
    semester_columns = [f'YR{yr}_{"FALL" if sem == 1 else "SPRING"}_LSU_GPA' for yr in range(1, year+1) for sem in range(1, 3)]
    return data[semester_columns].median(axis=1, skipna=True)

def fill_missing_gpas(data, year):
    """
    Fill missing LSU GPAs with the median LSU GPA of the available semesters based on the year of study.

    :param data: DataFrame containing LSU GPA columns.
    :param year: Year of study (1 for first year, 2 for second year, etc.)
    """
    median_gpas = calculate_median_gpas(data, year)
    semester_columns = [f'YR{yr}_{"FALL" if sem == 1 else "SPRING"}_LSU_GPA' for yr in range(1, year+1) for sem in range(1, 3)]

    for col in semester_columns:
        data[col] = data[col].fillna(median_gpas)

# Example usage
year_of_study = 2  
fill_missing_gpas(data, year_of_study)

# -----------------------------------------------------------------------------
# Handling the first year missing values for CUM_GPAs
# -----------------------------------------------------------------------------

def calculate_median_gpas(data, year):
    """
    Calculate the median cumulative GPA for each student based on the year of study.

    :param data: DataFrame containing cumulative GPA columns.
    :param year: Year of study (1 for first year, 2 for second year, etc.)
    :return: Series with median cumulative GPA for each student.
    """
    semester_columns = [f'YR{yr}_{"FALL" if sem == 1 else "SPRING"}_CUM_GPA' for yr in range(1, year+1) for sem in range(1, 3)]
    return data[semester_columns].median(axis=1, skipna=True)

def fill_missing_gpas(data, year):
    """
    Fill missing cumulative GPAs with the median cumulative GPA of the available semesters based on the year of study.

    :param data: DataFrame containing cumulative GPA columns.
    :param year: Year of study (1 for first year, 2 for second year, etc.)
    """
    median_gpas = calculate_median_gpas(data, year)
    semester_columns = [f'YR{yr}_{"FALL" if sem == 1 else "SPRING"}_CUM_GPA' for yr in range(1, year+1) for sem in range(1, 3)]

    for col in semester_columns:
        data[col] = data[col].fillna(median_gpas)

# Example usage
year_of_study = 1
fill_missing_gpas(data, year_of_study)

# -----------------------------------------------------------------------------
# Handling the second year missing values for CUM_GPAs
# -----------------------------------------------------------------------------

def calculate_median_gpas(data, year):
    """
    Calculate the median cumulative GPA for each student based on the year of study.

    :param data: DataFrame containing cumulative GPA columns.
    :param year: Year of study (1 for first year, 2 for second year, etc.)
    :return: Series with median cumulative GPA for each student.
    """
    semester_columns = [f'YR{yr}_{"FALL" if sem == 1 else "SPRING"}_CUM_GPA' for yr in range(1, year+1) for sem in range(1, 3)]
    return data[semester_columns].median(axis=1, skipna=True)

def fill_missing_gpas(data, year):
    """
    Fill missing cumulative GPAs with the median cumulative GPA of the available semesters based on the year of study.

    :param data: DataFrame containing cumulative GPA columns.
    :param year: Year of study (1 for first year, 2 for second year, etc.)
    """
    median_gpas = calculate_median_gpas(data, year)
    semester_columns = [f'YR{yr}_{"FALL" if sem == 1 else "SPRING"}_CUM_GPA' for yr in range(1, year+1) for sem in range(1, 3)]

    for col in semester_columns:
        data[col] = data[col].fillna(median_gpas)

# Example usage
year_of_study = 2  
fill_missing_gpas(data, year_of_study)

# -----------------------------------------------------------------------------
# Handling the first year missing values for cumulative hours carried
# -----------------------------------------------------------------------------

def calculate_median_hours(data, year):
    """
    Calculate the median cumulative hours carried for each student based on the year of study.

    :param data: DataFrame containing cumulative hours carried columns.
    :param year: Year of study (1 for first year, 2 for second year, etc.)
    :return: Series with median cumulative hours carried for each student.
    """
    semester_columns = [f'YR{yr}_{"FALL" if sem == 1 else "SPRING"}_CUM_HRS_CARR' for yr in range(1, year+1) for sem in range(1, 3)]
    return data[semester_columns].median(axis=1, skipna=True)

def fill_missing_hours(data, year):
    """
    Fill missing cumulative hours carried with the median cumulative hours carried of the available semesters based on the year of study.

    :param data: DataFrame containing cumulative hours carried columns.
    :param year: Year of study (1 for first year, 2 for second year, etc.)
    """
    median_hours = calculate_median_hours(data, year)
    semester_columns = [f'YR{yr}_{"FALL" if sem == 1 else "SPRING"}_CUM_HRS_CARR' for yr in range(1, year+1) for sem in range(1, 3)]

    for col in semester_columns:
        data[col] = data[col].fillna(median_hours)

# Example usage
year_of_study = 1
fill_missing_hours(data, year_of_study)

# -----------------------------------------------------------------------------
# Handling the second year missing values for cumulative hours carried
# -----------------------------------------------------------------------------

def calculate_median_hours(data, year):
    """
    Calculate the median cumulative hours carried for each student based on the year of study.

    :param data: DataFrame containing cumulative hours carried columns.
    :param year: Year of study (1 for first year, 2 for second year, etc.)
    :return: Series with median cumulative hours carried for each student.
    """
    semester_columns = [f'YR{yr}_{"FALL" if sem == 1 else "SPRING"}_CUM_HRS_CARR' for yr in range(1, year+1) for sem in range(1, 3)]
    return data[semester_columns].median(axis=1, skipna=True)

def fill_missing_hours(data, year):
    """
    Fill missing cumulative hours carried with the median cumulative hours carried of the available semesters based on the year of study.

    :param data: DataFrame containing cumulative hours carried columns.
    :param year: Year of study (1 for first year, 2 for second year, etc.)
    """
    median_hours = calculate_median_hours(data, year)
    semester_columns = [f'YR{yr}_{"FALL" if sem == 1 else "SPRING"}_CUM_HRS_CARR' for yr in range(1, year+1) for sem in range(1, 3)]

    for col in semester_columns:
        data[col] = data[col].fillna(median_hours)

# Example usage
year_of_study = 2  
fill_missing_hours(data, year_of_study)

# -----------------------------------------------------------------------------
# Handling the first year missing values for cumulative hours earned
# -----------------------------------------------------------------------------

def calculate_median_hours_earned(data, year):
    """
    Calculate the median cumulative hours earned for each student based on the year of study.

    :param data: DataFrame containing cumulative hours earned columns.
    :param year: Year of study (1 for first year, 2 for second year, etc.)
    :return: Series with median cumulative hours earned for each student.
    """
    semester_columns = [f'YR{yr}_{"FALL" if sem == 1 else "SPRING"}_CUM_HRS_EARN' for yr in range(1, year+1) for sem in range(1, 3)]
    return data[semester_columns].median(axis=1, skipna=True)

def fill_missing_hours_earned(data, year):
    """
    Fill missing cumulative hours earned with the median cumulative hours earned of the available semesters based on the year of study.

    :param data: DataFrame containing cumulative hours earned columns.
    :param year: Year of study (1 for first year, 2 for second year, etc.)
    """
    median_hours_earned = calculate_median_hours_earned(data, year)
    semester_columns = [f'YR{yr}_{"FALL" if sem == 1 else "SPRING"}_CUM_HRS_EARN' for yr in range(1, year+1) for sem in range(1, 3)]

    for col in semester_columns:
        data[col] = data[col].fillna(median_hours_earned)

# Example usage
year_of_study = 1
fill_missing_hours_earned(data, year_of_study)

# -----------------------------------------------------------------------------
# Handling the second year missing values for cumulative hours earned
# -----------------------------------------------------------------------------

def calculate_median_hours_earned(data, year):
    """
    Calculate the median cumulative hours earned for each student based on the year of study.

    :param data: DataFrame containing cumulative hours earned columns.
    :param year: Year of study (1 for first year, 2 for second year, etc.)
    :return: Series with median cumulative hours earned for each student.
    """
    semester_columns = [f'YR{yr}_{"FALL" if sem == 1 else "SPRING"}_CUM_HRS_EARN' for yr in range(1, year+1) for sem in range(1, 3)]
    return data[semester_columns].median(axis=1, skipna=True)

def fill_missing_hours_earned(data, year):
    """
    Fill missing cumulative hours earned with the median cumulative hours earned of the available semesters based on the year of study.

    :param data: DataFrame containing cumulative hours earned columns.
    :param year: Year of study (1 for first year, 2 for second year, etc.)
    """
    median_hours_earned = calculate_median_hours_earned(data, year)
    semester_columns = [f'YR{yr}_{"FALL" if sem == 1 else "SPRING"}_CUM_HRS_EARN' for yr in range(1, year+1) for sem in range(1, 3)]

    for col in semester_columns:
        data[col] = data[col].fillna(median_hours_earned)

# Example usage
year_of_study = 2  
fill_missing_hours_earned(data, year_of_study)

# -----------------------------------------------------------------------------
# Academic Status Encoding
# -----------------------------------------------------------------------------

# encode categorical data into numerical values
def encode_academic_status(data):
    """
    Encode the academic statuses as numerical values.

    :param data: DataFrame containing academic status columns.
    :return: DataFrame with encoded academic statuses.
    """
    status_mapping = {'Good': 3, 'Warn': 2, 'Prob': 1, 'Drop': 0}
    columns_to_encode = ['AC_ACT1', 'AC_ACT2', 'AC_ACT3', 'AC_ACT4']

    for col in columns_to_encode:
        data[col] = data[col].map(status_mapping)

    return data

# Encode the academic statuses
data = encode_academic_status(data)

# -----------------------------------------------------------------------------
# Handling the first year missing values for Academic_Status: AC_ACT
# -----------------------------------------------------------------------------

def calculate_median_statuses(data, year):
    """
    Calculate the median status for each student based on the year of study.

    :param data: DataFrame containing encoded academic status columns.
    :param year: Year of study (1, 2, etc.)
    :return: Series with median status for each student.
    """
    status_columns = [f'AC_ACT{i}' for i in range(1, year * 2 + 1)]
    return data[status_columns].median(axis=1, skipna=True)

def fill_missing_statuses(data, year):
    """
    Fill missing academic statuses with the median status based on the year of study.

    :param data: DataFrame containing encoded academic status columns.
    :param year: Year of study (1, 2, etc.)
    """
    median_statuses = calculate_median_statuses(data, year)
    status_columns = [f'AC_ACT{i}' for i in range(1, year * 2 + 1)]

    for col in status_columns:
        data[col] = data[col].fillna(median_statuses)

# Example Usage
year_of_study = 1 
fill_missing_statuses(data, year_of_study)

# -----------------------------------------------------------------------------
# Handling the second year missing values for Academic_Status: AC_ACT
# -----------------------------------------------------------------------------

def calculate_median_statuses(data, year):
    """
    Calculate the median status for each student based on the year of study.

    :param data: DataFrame containing encoded academic status columns.
    :param year: Year of study (1, 2, etc.)
    :return: Series with median status for each student.
    """
    status_columns = [f'AC_ACT{i}' for i in range(1, year * 2 + 1)]
    return data[status_columns].median(axis=1, skipna=True)

def fill_missing_statuses(data, year):
    """
    Fill missing academic statuses with the median status based on the year of study.

    :param data: DataFrame containing encoded academic status columns.
    :param year: Year of study (1, 2, etc.)
    """
    median_statuses = calculate_median_statuses(data, year)
    status_columns = [f'AC_ACT{i}' for i in range(1, year * 2 + 1)]

    for col in status_columns:
        data[col] = data[col].fillna(median_statuses)

# Example Usage
year_of_study = 2  
fill_missing_statuses(data, year_of_study)

# -----------------------------------------------------------------------------
# Missing Values Removal
# -----------------------------------------------------------------------------

# List of columns to check for missing values
columns_to_check = [
    'YR1_FALL_SEM_GPA', 'YR1_SPRING_SEM_GPA', 'YR2_FALL_SEM_GPA', 'YR2_SPRING_SEM_GPA',
    'YR1_FALL_LSU_GPA', 'YR1_SPRING_LSU_GPA', 'YR2_FALL_LSU_GPA', 'YR2_SPRING_LSU_GPA',
    'YR1_FALL_CUM_GPA', 'YR1_SPRING_CUM_GPA', 'YR2_FALL_CUM_GPA', 'YR2_SPRING_CUM_GPA',
    'YR1_FALL_CUM_HRS_CARR', 'YR1_SPRING_CUM_HRS_CARR', 'YR2_FALL_CUM_HRS_CARR', 'YR2_SPRING_CUM_HRS_CARR',
    'YR1_FALL_CUM_HRS_EARN', 'YR1_SPRING_CUM_HRS_EARN', 'YR2_FALL_CUM_HRS_EARN', 'YR2_SPRING_CUM_HRS_EARN',
    'AC_ACT1', 'AC_ACT2', 'AC_ACT3', 'AC_ACT4'
]

# Remove rows with missing values in any of the specified columns
data = data.dropna(subset=columns_to_check)

# -----------------------------------------------------------------------------
# Zip and Country Geocoding
# -----------------------------------------------------------------------------

# Preprocess HOMEZIPCODE: consider only the first 5 digits for 9 digit codes
def preprocess_zipcode(zipcode):
    zipcode_str = str(zipcode)
    if pd.notna(zipcode):
        if len(zipcode_str) == 9:
            return zipcode_str[:5]
    return zipcode_str

data['HOMEZIPCODE'] = data['HOMEZIPCODE'].apply(preprocess_zipcode)

# Convert HOMEZIPCODE to numeric, handling missing and non-numeric values
data['HOMEZIPCODE'] = pd.to_numeric(data['HOMEZIPCODE'], errors='coerce')

# Create a dictionary from the zip code coordinates dataset
zip_coordinates_dict = zip_coordinates.set_index('Zipcode')[['Latitude', 'Longitude']].to_dict(orient='index')

# Create a dictionary from the country coordinates dataset
country_coord_dict = country_coordinates.set_index('country')[['latitude', 'longitude']].to_dict(orient='index')

# Function to get coordinates based on ZIP code
def get_zip_coordinates(zipcode):
    if pd.isna(zipcode):
        return {'Latitude': None, 'Longitude': None}
    return zip_coordinates_dict.get(int(zipcode), {'Latitude': None, 'Longitude': None})

# Function to update coordinates based on country
def update_coordinates(row):
    if row['COUNTRY_DESC'] != 'United States':
        country_coords = country_coord_dict.get(row['COUNTRY_DESC'], {'latitude': None, 'longitude': None})
        return country_coords['latitude'], country_coords['longitude']
    return row['Latitude'], row['Longitude']

# Add initial Latitude and Longitude columns based on ZIP code
data['Latitude'] = data['HOMEZIPCODE'].apply(lambda x: get_zip_coordinates(x)['Latitude'])
data['Longitude'] = data['HOMEZIPCODE'].apply(lambda x: get_zip_coordinates(x)['Longitude'])

# Update Latitude and Longitude in the main dataset based on country
data[['Latitude', 'Longitude']] = data.apply(update_coordinates, axis=1, result_type='expand')

# Replace missing values in 'Latitude' and 'Longitude' with 0.00
data['Latitude'].fillna(0.00, inplace=True)
data['Longitude'].fillna(0.00, inplace=True)

# -----------------------------------------------------------------------------
# Financial estimation by zip for replacing missing values
# -----------------------------------------------------------------------------

# Calculate median EXP_FAM_CONTRIB and INCOME for each ZIP code
zip_code_medians = data.groupby('HOMEZIPCODE')[['EXP_FAM_CONTRIB', 'INCOME']].median()

# Calculate global median EXP_FAM_CONTRIB and INCOME for international students
global_medians_international = data[data['FOREIGN'] == 1][['EXP_FAM_CONTRIB', 'INCOME']].median()

# Function to estimate missing values based on ZIP code or global median for international students
def estimate_value(row, column):
    if pd.isnull(row[column]):
        if row['FOREIGN'] == 1:  
            return global_medians_international[column]
        elif row['HOMEZIPCODE'] in zip_code_medians.index: 
            return zip_code_medians.loc[row['HOMEZIPCODE'], column]
        else:
            return None  
    else:
        return row[column]

# Estimate missing values
for column in ['EXP_FAM_CONTRIB', 'INCOME']:
    data[column] = data.apply(lambda row: estimate_value(row, column), axis=1)

# Drop rows where EXP_FAM_CONTRIB or INCOME is still NaN after estimation
data.dropna(subset=['EXP_FAM_CONTRIB', 'INCOME'], inplace=True)

# -----------------------------------------------------------------------------
# Geographic Data Visualization
# -----------------------------------------------------------------------------

# Create a Basemap instance
map = Basemap(projection='merc', llcrnrlat=25, urcrnrlat=50, llcrnrlon=-125, urcrnrlon=-66)

# Draw coastlines and map boundaries
map.drawcoastlines()
map.drawmapboundary()

# Convert latitude and longitude to x and y coordinates
x, y = map(data['Longitude'].values, data['Latitude'].values)

# Plot using scatter, adjust size (s) and color (c) as needed
map.scatter(x, y, s=10, c='red', marker='o', alpha=0.5)

plt.show()

# Create a base map
map = folium.Map(location=[40, -95], zoom_start=4)  # Center of the US

# Ensure you're dropping NA values for latitude and longitude
heat_data = [[row['Latitude'], row['Longitude']] for index, row in data.dropna(subset=['Latitude', 'Longitude']).iterrows()]

# Add heat map layer
HeatMap(heat_data).add_to(map)

# Save or show the map
map.save('C:\\Users\\e_sho\\Desktop\\heatmap.html')
# or display in a Jupyter notebook
#map # Uncomment this if you're in a Jupyter notebook to display the map

# ----------------------------------------------------------------------------------------
# create a new column representing graduation or non-graduation. response variable (Y)
# ----------------------------------------------------------------------------------------

# Function to encode the GRAD_YRS values
def encode_grad_years(years):
    if 1 <= years <= 8:
        return 1
    else:
        return 0

# Apply the function to the GRAD_YRS column
data['GRAD_YRS_ENCODED'] = data['GRAD_YRS'].apply(lambda x: encode_grad_years(x) if pd.notnull(x) else 0)

# ----------------------------------------------------------------------------------------
# Selective column removal
# ----------------------------------------------------------------------------------------

# List of columns to drop
columns_to_drop = [
    'COHORTTRM', 'COHORTTRM_DESC', 'CH_ENRLTYPE', 'CHRTTYPE', 'CRSLOAD', 'LAST_TERM', 'LAST_TERM_DESC',
    'DEGR_TERM', 'DEGR_TERM_DESC', 'DEGR_PROGRAM', 'DEGR_CURRIC', 'DEGR_COLLEGE', 'DEGR_STEM', 'GRAD_YRS', 'GRAD_4',
    'GRAD_5', 'GRAD_6', 'GRAD_LAST_CUM_GPA', 'GRAD_LAST_LSU_GPA', 'PVT_SCHOOL_FLAG', 'PRIOR_INST_LEVEL',
    'PRIOR_INST_OVRL_GPA', 'AGE_YEARS', 'GENDER', 'FOREIGN', 'COUNTRY_DESC', 'FIRST_GEN', 'LA_RESIDENT', 'HOMEZIPCODE',
    'FIRST_GEN_FLAG', 'ATHLETE', 'VETERAN_STATUS', 'HS_TOP_10', 'HS_TOP_25', 'HS_TOP_50', 'HS_BOTTOM_25', 'HS_BOTTOM_50',
    'CH_CURRICULUM_1', 'CH_CURRIC_DESC', 'CH_COLLEGE_1', 'CH_CURRIC_COLLEGE', 'YR1_FALL_ON_CAMPUS', 'YR1_SPRING_ON_CAMPUS',
    'YR2_FALL_ON_CAMPUS', 'YR2_SPRING_ON_CAMPUS', 'YR3_FALL_ON_CAMPUS', 'YR3_SPRING_ON_CAMPUS', 'YR4_FALL_ON_CAMPUS',
    'YR4_SPRING_ON_CAMPUS', 'YR5_FALL_ON_CAMPUS', 'YR5_SPRING_ON_CAMPUS', 'YR6_FALL_ON_CAMPUS', 'YR6_SPRING_ON_CAMPUS',
    'YR7_FALL_ON_CAMPUS', 'YR7_SPRING_ON_CAMPUS', 'YR8_FALL_ON_CAMPUS', 'YR8_SPRING_ON_CAMPUS', 'YR3_FALL_SEM_GPA',
    'YR3_SPRING_SEM_GPA', 'YR4_FALL_SEM_GPA', 'YR4_SPRING_SEM_GPA', 'YR5_FALL_SEM_GPA', 'YR5_SPRING_SEM_GPA',
    'YR6_FALL_SEM_GPA', 'YR6_SPRING_SEM_GPA', 'YR7_FALL_SEM_GPA', 'YR7_SPRING_SEM_GPA', 'YR8_FALL_SEM_GPA',
    'YR8_SPRING_SEM_GPA', 'YR3_FALL_LSU_GPA', 'YR3_SPRING_LSU_GPA', 'YR4_FALL_LSU_GPA', 'YR4_SPRING_LSU_GPA',
    'YR5_FALL_LSU_GPA', 'YR5_SPRING_LSU_GPA', 'YR6_FALL_LSU_GPA', 'YR6_SPRING_LSU_GPA', 'YR7_FALL_LSU_GPA',
    'YR7_SPRING_LSU_GPA', 'YR8_FALL_LSU_GPA', 'YR8_SPRING_LSU_GPA', 'YR3_FALL_CUM_GPA', 'YR3_SPRING_CUM_GPA',
    'YR4_FALL_CUM_GPA', 'YR4_SPRING_CUM_GPA', 'YR5_FALL_CUM_GPA', 'YR5_SPRING_CUM_GPA', 'YR6_FALL_CUM_GPA',
    'YR6_SPRING_CUM_GPA', 'YR7_FALL_CUM_GPA', 'YR7_SPRING_CUM_GPA', 'YR8_FALL_CUM_GPA', 'YR8_SPRING_CUM_GPA',
    'YR3_FALL_CUM_HRS_CARR', 'YR3_SPRING_CUM_HRS_CARR', 'YR4_FALL_CUM_HRS_CARR', 'YR4_SPRING_CUM_HRS_CARR',
    'YR5_FALL_CUM_HRS_CARR', 'YR5_SPRING_CUM_HRS_CARR', 'YR6_FALL_CUM_HRS_CARR', 'YR6_SPRING_CUM_HRS_CARR',
    'YR7_FALL_CUM_HRS_CARR', 'YR7_SPRING_CUM_HRS_CARR', 'YR8_FALL_CUM_HRS_CARR', 'YR8_SPRING_CUM_HRS_CARR',
    'YR3_FALL_CUM_HRS_EARN', 'YR3_SPRING_CUM_HRS_EARN', 'YR4_FALL_CUM_HRS_EARN', 'YR4_SPRING_CUM_HRS_EARN',
    'YR5_FALL_CUM_HRS_EARN', 'YR5_SPRING_CUM_HRS_EARN', 'YR6_FALL_CUM_HRS_EARN', 'YR6_SPRING_CUM_HRS_EARN',
    'YR7_FALL_CUM_HRS_EARN', 'YR7_SPRING_CUM_HRS_EARN', 'YR8_FALL_CUM_HRS_EARN', 'YR8_SPRING_CUM_HRS_EARN',
    'AC_ACT5', 'AC_ACT6', 'AC_ACT7', 'AC_ACT8', 'AC_ACT9', 'AC_ACT10', 'AC_ACT11', 'AC_ACT12', 'AC_ACT13', 'AC_ACT14',
    'AC_ACT15', 'AC_ACT16', 'programNameOriginal', 'inst_degreeLevelFilter', 'loc_status', 'emp_wageAgeAdjMax',
    'emp_statusDetail', 'MostRecentCompany', 'Most_Recent_Start', 'Most_Recent_End', 'StartYear_All_Jobs', 'EndYear_All_Jobs'
]

# Drop the columns
data.drop(columns_to_drop, axis=1, inplace=True)

# Save the modified DataFrame to a new CSV file
data.to_csv('C:\\Users\\e_sho\\Desktop\\modified_SSWD_POOL_V1.csv', index=False)