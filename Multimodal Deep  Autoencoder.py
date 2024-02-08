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

#-----------------------------------------------------------------------------------------
# Reading and examining the dataset
#-----------------------------------------------------------------------------------------

# Reading the SSWD_POOL_V1 CSV file
data = pd.read_csv("modified_SSWD_POOL_V1.csv", low_memory=False)

# examine the dataset
print(data.info())

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
    'PVT_SCHOOL_FLAG_encoded', 'GENDER_encoded', 'FIRST_GEN_FLAG_encoded',
    'COUNTRY_DESC_encoded', 'GRAD_YRS_ENCODED', 'HS_SCORE', 'Cumulative_Campus_Presence_Score',
    'Program_AAAS', 'College_Agriculture', 'Curriculum_AAAS', 'College_ADSN', 'Curric_College_ADSN'
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
    'EXP_FAM_CONTRIB', 'INCOME', 'HS_SCORE', 'Cumulative_Campus_Presence_Score',
    'YR1_FALL_SEM_GPA', 'YR1_SPRING_SEM_GPA', 'YR2_FALL_SEM_GPA', 'YR2_SPRING_SEM_GPA',
    'YR1_FALL_LSU_GPA', 'YR1_SPRING_LSU_GPA', 'YR2_FALL_LSU_GPA', 'YR2_SPRING_LSU_GPA',
    'YR1_FALL_CUM_GPA', 'YR1_SPRING_CUM_GPA', 'YR2_FALL_CUM_GPA', 'YR2_SPRING_CUM_GPA',
    'YR1_FALL_CUM_HRS_CARR', 'YR1_SPRING_CUM_HRS_CARR', 'YR2_FALL_CUM_HRS_CARR',
    'YR2_SPRING_CUM_HRS_CARR','YR1_FALL_CUM_HRS_EARN', 'YR1_SPRING_CUM_HRS_EARN',
    'YR2_FALL_CUM_HRS_EARN', 'YR2_SPRING_CUM_HRS_EARN', 'AC_ACT1', 'AC_ACT2',
    'AC_ACT3', 'AC_ACT4'
]

# Calculate and print the range for each column
for col in columns_to_check:
    min_value = data[col].min()
    max_value = data[col].max()
    range_of_values = max_value - min_value
    print(f"Range of {col}: {range_of_values} (Min: {min_value}, Max: {max_value})")

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
# Create a histogram for each column: use a logarithmic scale for 'EXP_FAM_CONTRIB' and 'INCOME' for better visualization

for col in columns_to_plot:
    plt.figure(figsize=(10, 4))

    if col in ['EXP_FAM_CONTRIB', 'INCOME']:
        data_col = data[col].dropna()
        data_col = data_col.replace(0, np.nextafter(0, 1))
        plt.hist(data_col, bins=30, edgecolor='k', alpha=0.7, log=True)
        plt.xlabel(f'{col} (log scale)')
    else:
        plt.hist(data[col].dropna(), bins=30, edgecolor='k', alpha=0.7)
        plt.xlabel(col)

    plt.title(f'Histogram of {col}')
    plt.ylabel('Frequency')
    plt.show()

# List of categorical columns for which to plot pie charts
categorical_columns = [
    'HS_SCORE', 'Cumulative_Campus_Presence_Score', 'AC_ACT1', 'AC_ACT2', 'AC_ACT3', 'AC_ACT4',
    'PELL', 'TOPS', 'GREEK_ENTRY', 'CH_ENRLTYPE_encoded', 'CHRTTYPE_encoded', 'CRSLOAD_encoded',
    'PVT_SCHOOL_FLAG_encoded', 'GENDER_encoded', 'FIRST_GEN_FLAG_encoded', 'COUNTRY_DESC_encoded',
    'GRAD_YRS_ENCODED'
]

# Create a pie chart for each categorical column
for col in categorical_columns:
    plt.figure(figsize=(8, 8))
    data[col].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title(f'Pie Chart of {col}')
    plt.ylabel('')
    plt.show()

#------------------------------------------------------------------------------------------
# Data preparation
#------------------------------------------------------------------------------------------

# Dropping the 'PROJECT_ID' column
data = data.drop(columns=['PROJECT_ID'])

# Separating features and target variable
X = data.drop(columns=['GRAD_YRS_ENCODED'])
y = data['GRAD_YRS_ENCODED']

# Splitting the data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2

# Explicitly creating copies of the slices to avoid SettingWithCopyWarning
X_train = X_train.copy()
X_val = X_val.copy()
X_test = X_test.copy()

# Columns to be standardized
columns_to_standardize = [
    'BESTMATH', 'BESTENGL', 'BESTCOMP', 'HS_ACAD_AVG', 'HS_OVERALL_AVG', 'AGE_MONTHS',
    'EXP_FAM_CONTRIB', 'INCOME', 'YR1_FALL_SEM_GPA', 'YR1_SPRING_SEM_GPA', 'YR2_FALL_SEM_GPA',
    'YR2_SPRING_SEM_GPA', 'YR1_FALL_LSU_GPA', 'YR1_SPRING_LSU_GPA', 'YR2_FALL_LSU_GPA',
    'YR2_SPRING_LSU_GPA', 'YR1_FALL_CUM_GPA', 'YR1_SPRING_CUM_GPA', 'YR2_FALL_CUM_GPA',
    'YR2_SPRING_CUM_GPA', 'YR1_FALL_CUM_HRS_CARR', 'YR1_SPRING_CUM_HRS_CARR', 'YR2_FALL_CUM_HRS_CARR',
    'YR2_SPRING_CUM_HRS_CARR', 'YR1_FALL_CUM_HRS_EARN', 'YR1_SPRING_CUM_HRS_EARN', 'YR2_FALL_CUM_HRS_EARN',
    'YR2_SPRING_CUM_HRS_EARN', 'AC_ACT1', 'AC_ACT2', 'AC_ACT3', 'AC_ACT4', 'Latitude', 'Longitude',
    'HS_SCORE', 'Cumulative_Campus_Presence_Score'
]

# Standardizing the specified features in the training set
scaler = StandardScaler()
X_train[columns_to_standardize] = scaler.fit_transform(X_train[columns_to_standardize])

# Applying the same transformation to the validation and test sets
X_val[columns_to_standardize] = scaler.transform(X_val[columns_to_standardize])
X_test[columns_to_standardize] = scaler.transform(X_test[columns_to_standardize])

#-----------------------------------------
# plot histograms after standardization
#-----------------------------------------
'''
# List of numerical columns for which to plot histograms after standardization
numerical_columns = [
    'BESTMATH', 'BESTENGL', 'BESTCOMP', 'HS_ACAD_AVG', 'HS_OVERALL_AVG', 'AGE_MONTHS',
    'EXP_FAM_CONTRIB', 'INCOME', 'HS_SCORE', 'Cumulative_Campus_Presence_Score',
    'YR1_FALL_SEM_GPA', 'YR1_SPRING_SEM_GPA', 'YR2_FALL_SEM_GPA', 'YR2_SPRING_SEM_GPA',
    'YR1_FALL_LSU_GPA', 'YR1_SPRING_LSU_GPA', 'YR2_FALL_LSU_GPA', 'YR2_SPRING_LSU_GPA',
    'YR1_FALL_CUM_GPA', 'YR1_SPRING_CUM_GPA', 'YR2_FALL_CUM_GPA', 'YR2_SPRING_CUM_GPA',
    'YR1_FALL_CUM_HRS_CARR', 'YR1_SPRING_CUM_HRS_CARR', 'YR2_FALL_CUM_HRS_CARR', 'YR2_SPRING_CUM_HRS_CARR',
    'YR1_FALL_CUM_HRS_EARN', 'YR1_SPRING_CUM_HRS_EARN', 'YR2_FALL_CUM_HRS_EARN', 'YR2_SPRING_CUM_HRS_EARN'
]

# Create a histogram for each column after standardization
for col in numerical_columns:
    plt.figure(figsize=(10, 4))
    plt.hist(X_train[col].dropna(), bins=30, edgecolor='k', alpha=0.7)
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
    if col in X_train.columns:
        plt.figure(figsize=(8, 8))
        X_train[col].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title(f'Pie Chart of {col} (After Normalization)')
        plt.ylabel('')
        plt.show()
'''

#------------------------------------------------------------------------------------------
# Data preparation continuing
#------------------------------------------------------------------------------------------

# Function to convert Excel-style column header to index
def excel_column_to_index(col):
    index = 0
    for c in col:
        index = index * 26 + (ord(c.upper()) - ord('A')) + 1
    return index - 1

# Function to select columns based on Excel-style range (inclusive), adjusted for dropped column
def select_columns(dataframe, start_col, end_col):
    start_index = excel_column_to_index(start_col) - 1  # Adjusted for dropped column
    end_index = excel_column_to_index(end_col)  # No need to add 1, as end index is inclusive
    return dataframe.iloc[:, start_index:end_index]

# Selecting columns for different categories
program_data = select_columns(data, 'AR', 'DX')
college_data = select_columns(data, 'DY', 'EH')
curriculum_1_data = select_columns(data, 'EI', 'JR')
college_1_data = select_columns(data, 'JS', 'KE')
curric_college_data = select_columns(data, 'KF', 'KR')

#--------------------------
# Grouping the Features
#--------------------------

# Group 1: Program Data
program_data_train = select_columns(X_train, 'AR', 'DX')
program_data_val = select_columns(X_val, 'AR', 'DX')
program_data_test = select_columns(X_test, 'AR', 'DX')

# Group 2: College Data
college_data_train = select_columns(X_train, 'DY', 'EH')
college_data_val = select_columns(X_val, 'DY', 'EH')
college_data_test = select_columns(X_test, 'DY', 'EH')

# Group 3: Curriculum 1 Data
curriculum_1_data_train = select_columns(X_train, 'EI', 'JR')
curriculum_1_data_val = select_columns(X_val, 'EI', 'JR')
curriculum_1_data_test = select_columns(X_test, 'EI', 'JR')

# Group 4: College 1 Data
college_1_data_train = select_columns(X_train, 'JS', 'KE')
college_1_data_val = select_columns(X_val, 'JS', 'KE')
college_1_data_test = select_columns(X_test, 'JS', 'KE')

# Group 5: Curric College Data
curric_college_data_train = select_columns(X_train, 'KF', 'KR')
curric_college_data_val = select_columns(X_val, 'KF', 'KR')
curric_college_data_test = select_columns(X_test, 'KF', 'KR')

# Group 6: Other Features
other_features_columns = [
    'BESTMATH', 'BESTENGL', 'BESTCOMP', 'HS_ACAD_AVG', 'HS_OVERALL_AVG', 'AGE_MONTHS',
    'EXP_FAM_CONTRIB', 'INCOME', 'PELL', 'TOPS', 'GREEK_ENTRY', 'YR1_FALL_SEM_GPA',
    'YR1_SPRING_SEM_GPA', 'YR2_FALL_SEM_GPA', 'YR2_SPRING_SEM_GPA', 'YR1_FALL_LSU_GPA',
    'YR1_SPRING_LSU_GPA', 'YR2_FALL_LSU_GPA', 'YR2_SPRING_LSU_GPA', 'YR1_FALL_CUM_GPA',
    'YR1_SPRING_CUM_GPA', 'YR2_FALL_CUM_GPA', 'YR2_SPRING_CUM_GPA', 'YR1_FALL_CUM_HRS_CARR',
    'YR1_SPRING_CUM_HRS_CARR', 'YR2_FALL_CUM_HRS_CARR', 'YR2_SPRING_CUM_HRS_CARR',
    'YR1_FALL_CUM_HRS_EARN', 'YR1_SPRING_CUM_HRS_EARN', 'YR2_FALL_CUM_HRS_EARN',
    'YR2_SPRING_CUM_HRS_EARN', 'AC_ACT1', 'AC_ACT2', 'AC_ACT3', 'AC_ACT4', 'Latitude',
    'Longitude', 'CH_ENRLTYPE_encoded', 'CHRTTYPE_encoded', 'CRSLOAD_encoded',
    'PVT_SCHOOL_FLAG_encoded', 'GENDER_encoded', 'FIRST_GEN_FLAG_encoded',
    'COUNTRY_DESC_encoded', 'HS_SCORE', 'Cumulative_Campus_Presence_Score'
]

other_features_train = X_train[other_features_columns]
other_features_val = X_val[other_features_columns]
other_features_test = X_test[other_features_columns]

#-------------------------------------------------
# Convert the grouped features to PyTorch tensors
#-------------------------------------------------

# Group 1: Program Data
program_data_train_tensor = torch.tensor(program_data_train.values, dtype=torch.float32)
program_data_val_tensor = torch.tensor(program_data_val.values, dtype=torch.float32)
program_data_test_tensor = torch.tensor(program_data_test.values, dtype=torch.float32)

# Group 2: College Data
college_data_train_tensor = torch.tensor(college_data_train.values, dtype=torch.float32)
college_data_val_tensor = torch.tensor(college_data_val.values, dtype=torch.float32)
college_data_test_tensor = torch.tensor(college_data_test.values, dtype=torch.float32)

# Group 3: Curriculum 1 Data
curriculum_1_data_train_tensor = torch.tensor(curriculum_1_data_train.values, dtype=torch.float32)
curriculum_1_data_val_tensor = torch.tensor(curriculum_1_data_val.values, dtype=torch.float32)
curriculum_1_data_test_tensor = torch.tensor(curriculum_1_data_test.values, dtype=torch.float32)

# Group 4: College 1 Data
college_1_data_train_tensor = torch.tensor(college_1_data_train.values, dtype=torch.float32)
college_1_data_val_tensor = torch.tensor(college_1_data_val.values, dtype=torch.float32)
college_1_data_test_tensor = torch.tensor(college_1_data_test.values, dtype=torch.float32)

# Group 5: Curric College Data
curric_college_data_train_tensor = torch.tensor(curric_college_data_train.values, dtype=torch.float32)
curric_college_data_val_tensor = torch.tensor(curric_college_data_val.values, dtype=torch.float32)
curric_college_data_test_tensor = torch.tensor(curric_college_data_test.values, dtype=torch.float32)

# Group 6: Other Features
other_features_train_tensor = torch.tensor(other_features_train.values, dtype=torch.float32)
other_features_val_tensor = torch.tensor(other_features_val.values, dtype=torch.float32)
other_features_test_tensor = torch.tensor(other_features_test.values, dtype=torch.float32)

# Convert the target variable to PyTorch tensors
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

#-------------------
# Dataset loading
#-------------------

# Batch size definition
batch_size = 32

# Create TensorDataset objects
train_dataset = TensorDataset(program_data_train_tensor, college_data_train_tensor, curriculum_1_data_train_tensor, college_1_data_train_tensor, curric_college_data_train_tensor, other_features_train_tensor, y_train_tensor)
val_dataset = TensorDataset(program_data_val_tensor, college_data_val_tensor, curriculum_1_data_val_tensor, college_1_data_val_tensor, curric_college_data_val_tensor, other_features_val_tensor, y_val_tensor)
test_dataset = TensorDataset(program_data_test_tensor, college_data_test_tensor, curriculum_1_data_test_tensor, college_1_data_test_tensor, curric_college_data_test_tensor, other_features_test_tensor, y_test_tensor)

# Create DataLoader objects
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#--------------------------------------------------------------------------------------------------
# First Architecture Multimodal Deep Autoencoder
#--------------------------------------------------------------------------------------------------

#----------------------
# Autoencoder Sizing
#----------------------

# Input sizes for each feature group
input_sizes = [85, 10, 140, 13, 13, 46]  # Number of features in each group

# Encoding sizes (example: half of the input sizes)
#encoding_sizes = [size // 2 for size in input_sizes]

# Adjusted encoding sizes - example: you can adjust this based on your experiment
encoding_sizes = [10, 2, 15, 3, 3, 6]

# Decoding sizes (same as encoding sizes for a symmetric autoencoder)
decoding_sizes = encoding_sizes.copy()

#-----------------------------------
# Multi-Input Autoencoder Design
#-----------------------------------

class MultiInputAutoencoder(nn.Module):
    def __init__(self, input_sizes, encoding_sizes, decoding_sizes, central_size=256):
        super(MultiInputAutoencoder, self).__init__()
        self.input_sizes = input_sizes

        # Encode layers for each input
        self.encoders = nn.ModuleList(
            [nn.Linear(in_size, en_size) for in_size, en_size in zip(input_sizes, encoding_sizes)])

        # Central encoder and decoder with configurable size
        self.central_encoder = nn.Sequential(nn.Linear(sum(encoding_sizes), central_size), nn.ReLU())
        self.central_decoder = nn.Sequential(nn.Linear(central_size, sum(decoding_sizes)), nn.ReLU())

        # Decode layers for each output
        self.decoders = nn.ModuleList(
            [nn.Linear(de_size, in_size) for de_size, in_size in zip(decoding_sizes, input_sizes)])

    def validate_input_sizes(self, inputs):
        for input_tensor, expected_size in zip(inputs, self.input_sizes):
            if input_tensor.shape[1] != expected_size:
                raise ValueError(f"Expected input size {expected_size}, got {input_tensor.shape[1]}")

    def forward(self, inputs):
        # Validate input sizes
        self.validate_input_sizes(inputs)

        # Encoding each input
        encoded_inputs = [encoder(input) for encoder, input in zip(self.encoders, inputs)]
        concatenated = torch.cat(encoded_inputs, dim=1)

        # Central encoding and decoding
        encoded = self.central_encoder(concatenated)
        decoded = self.central_decoder(encoded)

        # Splitting the decoded output
        outputs = []
        start = 0
        for decoder, size in zip(self.decoders, decoding_sizes):
            outputs.append(decoder(decoded[:, start:start + size]))
            start += size

        return outputs

# Create the model
model = MultiInputAutoencoder(input_sizes, encoding_sizes, decoding_sizes)

# Define the EarlyStopping Class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Training Function
def train_model(model, train_loader, val_loader, criterion, num_epochs=100, learning_rate=0.001):
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    training_losses = []
    validation_losses = []
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    for epoch in range(num_epochs):
        total_loss = 0

        for batch in train_loader:
            # Unpack the batch. The last element is the target, and the rest are inputs
            *inputs, targets = batch

            # Validate input sizes
            model.validate_input_sizes(inputs)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            reconstructed = model(inputs)

            # Calculate loss
            loss = sum(criterion(recon, orig) for recon, orig in zip(reconstructed, inputs))

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate and store training loss
        average_training_loss = total_loss / len(train_loader)
        training_losses.append(average_training_loss)

        # Evaluate on validation set and store the loss
        average_validation_loss = evaluate_model_per_epoch(model, val_loader, criterion)
        validation_losses.append(average_validation_loss)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {average_training_loss:.4f}, Validation Loss: {average_validation_loss:.4f}')

        # Early stopping check
        early_stopping(average_validation_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    return training_losses, validation_losses

#  Evaluation Function
def evaluate_model_per_epoch(model, val_loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            *inputs, _ = batch
            reconstructed = model(inputs)
            loss = sum(criterion(recon, orig) for recon, orig in zip(reconstructed, inputs))
            total_loss += loss.item()

    average_loss = total_loss / len(val_loader)
    return average_loss

# Define the number of epochs and learning rate
num_epochs = 100
learning_rate = 0.0001

# Define the loss function
criterion = nn.MSELoss()

# Train the model
training_losses, validation_losses = train_model(model, train_loader, val_loader, criterion, num_epochs, learning_rate)

# Plot the Training and Validation Losses
plt.figure(figsize=(10, 6))
plt.plot(training_losses, label='Training Loss', color='blue')
plt.plot(validation_losses, label='Validation Loss', color='red')

plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the Model on the Test Set
def evaluate_model_on_test(model, test_loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            *inputs, _ = batch
            reconstructed = model(inputs)
            loss = sum(criterion(recon, orig) for recon, orig in zip(reconstructed, inputs))
            total_loss += loss.item()

    average_loss = total_loss / len(test_loader)
    print(f'Average Test Loss: {average_loss:.4f}')

# Evaluate on test set
evaluate_model_on_test(model, test_loader, criterion)

#---------------------------
# t-SNE visualization
#---------------------------
def extract_encoded_features(model, data_loader):
    model.eval()
    encoded_features = []
    y_values = []

    with torch.no_grad():
        for batch in data_loader:
            *inputs, labels = batch
            encoded_inputs = [model.encoders[i](input) for i, input in enumerate(inputs)]
            concatenated = torch.cat(encoded_inputs, dim=1)
            encoded = model.central_encoder(concatenated)
            encoded_flat = encoded.view(encoded.size(0), -1)
            encoded_features.append(encoded_flat.cpu().numpy())
            y_values.extend(labels.cpu().numpy())

    # Concatenate all features and labels into a single NumPy array
    encoded_features = np.concatenate(encoded_features, axis=0)
    y_values = np.array(y_values)  # Convert y_values to a NumPy array
    return encoded_features, y_values

# Extract encoded features and labels
encoded_test_features, y_test = extract_encoded_features(model, test_loader)

# Apply t-SNE to the encoded features
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
tsne_results = tsne.fit_transform(encoded_test_features)

# Visualization with response variable
plt.figure(figsize=(10, 6))
for class_value in np.unique(y_test):
    indices = np.where(y_test == class_value)
    plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=f'Class {class_value}')
plt.xlabel('t-SNE feature 0')
plt.ylabel('t-SNE feature 1')
plt.title('t-SNE Visualization of Encoded Features with Response Variable')
plt.legend()
plt.show()

#--------------------------------------------------------------------------------------------------
# Second Architecture Multimodal Deep Autoencoder
#--------------------------------------------------------------------------------------------------

class ModalityEncoder(nn.Module):
    def __init__(self, input_size, encoded_size):
        super(ModalityEncoder, self).__init__()
        self.encoder = nn.Linear(input_size, encoded_size)

    def forward(self, x):
        return self.encoder(x)

class ModalityDecoder(nn.Module):
    def __init__(self, encoded_size, output_size):
        super(ModalityDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(encoded_size, output_size),
            nn.Linear(output_size, output_size)
        )

    def forward(self, x):
        return self.decoder(x)

class MultimodalAutoencoder(nn.Module):
    def __init__(self, input_sizes, encoded_sizes, central_size, transform_size):
        super(MultimodalAutoencoder, self).__init__()
        self.encoders = nn.ModuleList(
            [ModalityEncoder(input_size, encoded_size) for input_size, encoded_size in zip(input_sizes, encoded_sizes)])

        # Layers after initial concatenation
        self.post_concat_transform = nn.Sequential(
            nn.Linear(sum(encoded_sizes), transform_size),
            nn.ReLU(),
            nn.Linear(transform_size, transform_size)
        )

        self.central_encoder = nn.Linear(transform_size, central_size)
        self.central_decoder = nn.Linear(central_size, transform_size)

        # Layers before final decoding
        self.pre_decode_transform = nn.Sequential(
            nn.Linear(transform_size, transform_size),
            nn.ReLU(),
            nn.Linear(transform_size, sum(encoded_sizes))
        )

        self.decoders = nn.ModuleList(
            [ModalityDecoder(encoded_size, input_size) for encoded_size, input_size in zip(encoded_sizes, input_sizes)])

    def encode(self, inputs):
        encoded_outputs = [encoder(input) for encoder, input in zip(self.encoders, inputs)]
        concatenated = torch.cat(encoded_outputs, dim=1)
        post_concat = self.post_concat_transform(concatenated)
        central_encoded = self.central_encoder(post_concat)
        return central_encoded

    def forward(self, inputs):
        encoded = [encoder(input) for encoder, input in zip(self.encoders, inputs)]
        concatenated = torch.cat(encoded, dim=1)
        post_concat = self.post_concat_transform(concatenated)
        central_encoded = self.central_encoder(post_concat)
        central_decoded = self.central_decoder(central_encoded)
        pre_decode = self.pre_decode_transform(central_decoded)

        split_sizes = [encoder.encoder.out_features for encoder in self.encoders]
        split_decoded = torch.split(pre_decode, split_sizes, dim=1)
        decoded = [decoder(split) for decoder, split in zip(self.decoders, split_decoded)]
        return decoded

# Example Usage
input_sizes = [85, 10, 140, 13, 13, 46]  # Input sizes for each group. This is fixed
encoded_sizes = [10, 2, 15, 3, 3, 6]  # Example encoded sizes (adjust as needed)
central_size = 15  # Example size of the central encoded representation (adjust as needed)
transform_size = 20  # Example size for transformation layers (adjust as needed)

model = MultimodalAutoencoder(input_sizes, encoded_sizes, central_size, transform_size)

# Example input data for each group
X1 = torch.randn(32, 85)  # Group 1: Program Data
X2 = torch.randn(32, 10)  # Group 2: College Data
X3 = torch.randn(32, 140) # Group 3: Curriculum 1 Data
X4 = torch.randn(32, 13)  # Group 4: College 1 Data
X5 = torch.randn(32, 13)  # Group 5: Curric College Data
X6 = torch.randn(32, 46)  # Group 6: Other Features

# Forward pass
outputs = model([X1, X2, X3, X4, X5, X6])


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def train_model(model, train_loader, val_loader, criterion, num_epochs=100, learning_rate=0.001):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    training_losses = []
    validation_losses = []
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    for epoch in range(num_epochs):
        total_loss = 0

        for batch in train_loader:
            *inputs, _ = batch
            inputs = [input.to(device) for input in inputs]

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = sum([criterion(output, input) for output, input in zip(outputs, inputs)])  # Reconstruction loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_training_loss = total_loss / len(train_loader)
        training_losses.append(average_training_loss)

        average_validation_loss = evaluate_model_per_epoch(model, val_loader, criterion)
        validation_losses.append(average_validation_loss)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {average_training_loss:.4f}, Validation Loss: {average_validation_loss:.4f}')

        early_stopping(average_validation_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    return training_losses, validation_losses

def evaluate_model_per_epoch(model, val_loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            *inputs, _ = batch  
            inputs = [input.to(device) for input in inputs]

            outputs = model(inputs)
            loss = sum([criterion(output, input) for output, input in zip(outputs, inputs)])  # Reconstruction loss

            total_loss += loss.item()

    average_loss = total_loss / len(val_loader)
    return average_loss

# Define the number of epochs and learning rate
num_epochs = 100
learning_rate = 0.0001

# Define the loss function
criterion = nn.MSELoss()

# Train the model
training_losses, validation_losses = train_model(model, train_loader, val_loader, criterion, num_epochs, learning_rate)

# Plot the Training and Validation Losses
plt.figure(figsize=(10, 6))
plt.plot(training_losses, label='Training Loss', color='blue')
plt.plot(validation_losses, label='Validation Loss', color='red')
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate on test set
def evaluate_model_on_test(model, test_loader, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            *inputs, _ = batch
            inputs = [input.to(device) for input in inputs]
            reconstructed = model(inputs)
            loss = sum(criterion(recon, orig) for recon, orig in zip(reconstructed, inputs))
            total_loss += loss.item()

    average_loss = total_loss / len(test_loader)
    print(f'Average Test Loss: {average_loss:.4f}')

evaluate_model_on_test(model, test_loader, criterion)

#---------------------------
# t-SNE visualization
#---------------------------

def extract_encoded_features(model, data_loader):
    model.eval()
    encoded_features = []
    y_values = []

    with torch.no_grad():
        for batch in data_loader:
            *inputs, labels = batch
            inputs = [input.to(device) for input in inputs]

            encoded = model.encode(inputs)
            encoded_flat = encoded.view(encoded.size(0), -1)
            encoded_features.append(encoded_flat.cpu().numpy())
            y_values.extend(labels.cpu().numpy())

    encoded_features = np.concatenate(encoded_features, axis=0)
    y_values = np.array(y_values)
    return encoded_features, y_values

# Extract encoded features and labels
encoded_test_features, y_test = extract_encoded_features(model, test_loader)

# Apply t-SNE to the encoded features
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
tsne_results = tsne.fit_transform(encoded_test_features)

# Visualization with response variable
plt.figure(figsize=(10, 6))
for class_value in np.unique(y_test):
    indices = np.where(y_test == class_value)
    plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=f'Class {class_value}')
plt.xlabel('t-SNE feature 0')
plt.ylabel('t-SNE feature 1')
plt.title('t-SNE Visualization of Encoded Features with Response Variable')
plt.legend()
plt.show()

#--------------------------------------------------------------------------------------------------
# Third Architecture Multimodal Deep Autoencoder with classifier
#--------------------------------------------------------------------------------------------------

class ModalityEncoder(nn.Module):
    def __init__(self, input_size, encoded_size):
        super(ModalityEncoder, self).__init__()
        self.encoder = nn.Linear(input_size, encoded_size)

    def forward(self, x):
        return self.encoder(x)

class ModalityDecoder(nn.Module):
    def __init__(self, encoded_size, output_size):
        super(ModalityDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(encoded_size, output_size),
            nn.Linear(output_size, output_size)
        )

    def forward(self, x):
        return self.decoder(x)

class MultimodalAutoencoderWithClassifier(nn.Module):
    def __init__(self, input_sizes, encoded_sizes, central_size, transform_size, classifier_output_size=1):
        super(MultimodalAutoencoderWithClassifier, self).__init__()
        self.encoders = nn.ModuleList(
            [ModalityEncoder(input_size, encoded_size) for input_size, encoded_size in zip(input_sizes, encoded_sizes)])
        self.post_concat_transform = nn.Sequential(
            nn.Linear(sum(encoded_sizes), transform_size),
            nn.ReLU(),
            nn.Linear(transform_size, transform_size)
        )
        self.central_encoder = nn.Linear(transform_size, central_size)
        self.central_decoder = nn.Linear(central_size, transform_size)
        self.pre_decode_transform = nn.Sequential(
            nn.Linear(transform_size, transform_size),
            nn.ReLU(),
            nn.Linear(transform_size, sum(encoded_sizes))
        )
        self.decoders = nn.ModuleList(
            [ModalityDecoder(encoded_size, input_size) for encoded_size, input_size in zip(encoded_sizes, input_sizes)])

        self.classifier = nn.Sequential(
            nn.Linear(central_size, 128),  # Example size, adjust as needed
            nn.ReLU(),
            nn.Dropout(0.5),  # Example dropout, adjust as needed
            nn.Linear(128, classifier_output_size),
            nn.Sigmoid()  # For binary classification
        )

    def forward(self, inputs):
        encoded = [encoder(input) for encoder, input in zip(self.encoders, inputs)]
        concatenated = torch.cat(encoded, dim=1)
        post_concat = self.post_concat_transform(concatenated)
        central_encoded = self.central_encoder(post_concat)

        central_decoded = self.central_decoder(central_encoded)
        pre_decode = self.pre_decode_transform(central_decoded)
        split_sizes = [encoder.encoder.out_features for encoder in self.encoders]
        split_decoded = torch.split(pre_decode, split_sizes, dim=1)
        decoded = [decoder(split) for decoder, split in zip(self.decoders, split_decoded)]

        classification_output = self.classifier(central_encoded)

        return *decoded, classification_output

    def encode(self, inputs):
        encoded = [self.encoders[i](inputs[i]) for i in range(len(inputs))]
        concatenated = torch.cat(encoded, dim=1)
        post_concat = self.post_concat_transform(concatenated)
        central_encoded = self.central_encoder(post_concat)
        return central_encoded

# Define the model parameters
input_sizes = [85, 10, 140, 13, 13, 46]
encoded_sizes = [30, 5, 40, 7, 7, 18]  # Adjust as needed
central_size = 50
transform_size = 80
classifier_output_size = 1

# Initialize the model
model = MultimodalAutoencoderWithClassifier(input_sizes, encoded_sizes, central_size, transform_size, classifier_output_size)

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare example input data
X1 = torch.randn(32, 85).to(device)  # Group 1: Program Data
X2 = torch.randn(32, 10).to(device)  # Group 2: College Data
X3 = torch.randn(32, 140).to(device) # Group 3: Curriculum 1 Data
X4 = torch.randn(32, 13).to(device)  # Group 4: College 1 Data
X5 = torch.randn(32, 13).to(device)  # Group 5: Curric College Data
X6 = torch.randn(32, 46).to(device)  # Group 6: Other Features

# Forward pass through the model
model.eval()
with torch.no_grad():
    outputs = model([X1, X2, X3, X4, X5, X6])
    *reconstructions, classification_output = outputs

    # Check the shapes of the outputs
    #print("Reconstructions Shapes:", [recon.shape for recon in reconstructions])
    #print("Classification Output Shape:", classification_output.shape)

# Define the EarlyStopping class
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

model.to(device)

# Define loss functions
criterion_reconstruction = nn.MSELoss()
criterion_classification = nn.BCELoss()

# Define an optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Example DataLoader setup
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

def train_model(model, train_loader, val_loader, criterion_reconstruction, criterion_classification, optimizer,
                num_epochs=100):
    training_losses = []
    validation_losses = []
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            *inputs, labels = batch
            inputs = [input.to(device) for input in inputs]
            labels = labels.to(device).float()

            optimizer.zero_grad()

            *reconstructions, classification_output = model(inputs)
            reconstruction_loss = sum(
                criterion_reconstruction(recon, input) for recon, input in zip(reconstructions, inputs))
            classification_loss = criterion_classification(classification_output.squeeze(), labels)
            total_loss = reconstruction_loss + classification_loss

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()

        avg_train_loss = train_loss / len(train_loader)
        training_losses.append(avg_train_loss)

        # Validation loss calculation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                *inputs, labels = batch
                inputs = [input.to(device) for input in inputs]
                labels = labels.to(device).float()

                *reconstructions, classification_output = model(inputs)
                reconstruction_loss = sum(
                    criterion_reconstruction(recon, input) for recon, input in zip(reconstructions, inputs))
                classification_loss = criterion_classification(classification_output.squeeze(), labels)
                total_val_loss = reconstruction_loss + classification_loss

                val_loss += total_val_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)

        print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        # Early Stopping check
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return training_losses, validation_losses

# Call the train_model function and plot the losses
training_losses, validation_losses = train_model(model, train_loader, val_loader, criterion_reconstruction,
                                                 criterion_classification, optimizer, num_epochs=100)

# Plot the Training and Validation Losses
plt.figure(figsize=(10, 5))
plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.title('Training and Validation Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Model evaluation
def calculate_test_loss(model, test_loader, criterion_reconstruction, criterion_classification):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            *inputs, labels = batch
            inputs = [input.to(device) for input in inputs]
            labels = labels.to(device).float()
            *reconstructions, classification_output = model(inputs)
            reconstruction_loss = sum(criterion_reconstruction(recon, inputs[i]) for i, recon in enumerate(reconstructions))
            classification_loss = criterion_classification(classification_output.squeeze(), labels)
            test_loss += reconstruction_loss.item() + classification_loss.item()
    print(f'Average Test Loss: {test_loss / len(test_loader):.4f}')

calculate_test_loss(model, test_loader, criterion_reconstruction, criterion_classification)

# Model accuracy calculation
def evaluate_performance(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            *inputs, labels = batch
            inputs = [input.to(device) for input in inputs]
            labels = labels.to(device).float()
            outputs = model(inputs)
            classification_output = outputs[-1]
            predicted = (classification_output.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the model on the test data: {100 * correct / total:.2f}%')

evaluate_performance(model, test_loader)

#-------------------------
# t-SNE visualization
#-------------------------

def extract_features_and_visualize(model, data_loader):
    model.eval()
    features = []
    labels_list = []
    with torch.no_grad():
        for batch in data_loader:
            *inputs, labels = batch
            inputs = [input.to(device) for input in inputs]
            encoded_features = model.encode(inputs).cpu().numpy()
            features.append(encoded_features)
            labels_list.append(labels.cpu().numpy())

    features = np.concatenate(features)
    labels_list = np.concatenate(labels_list)

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(10, 6))
    for class_value in np.unique(labels_list):
        indices = labels_list == class_value
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=f'Class {class_value}')
    plt.xlabel('t-SNE feature 0')
    plt.ylabel('t-SNE feature 1')
    plt.title('t-SNE Visualization of Encoded Features')
    plt.legend()
    plt.show()

extract_features_and_visualize(model, test_loader)

#--------------------------------------------------------------------------------------------------
# Fourth Architecture Multimodal Deep Autoencoder with classifier
#--------------------------------------------------------------------------------------------------

# Apply dropout, L1/L2 regularization, and batch normalization to the third architecture

class ModalityEncoder(nn.Module):
    def __init__(self, input_size, encoded_size):
        super(ModalityEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, encoded_size),
            nn.BatchNorm1d(encoded_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.encoder(x)

class ModalityDecoder(nn.Module):
    def __init__(self, encoded_size, output_size):
        super(ModalityDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(encoded_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.decoder(x)

class MultimodalAutoencoderWithClassifier(nn.Module):
    def __init__(self, input_sizes, encoded_sizes, central_size, transform_size, classifier_output_size=1):
        super(MultimodalAutoencoderWithClassifier, self).__init__()
        self.encoders = nn.ModuleList(
            [ModalityEncoder(input_size, encoded_size) for input_size, encoded_size in zip(input_sizes, encoded_sizes)])
        self.post_concat_transform = nn.Sequential(
            nn.Linear(sum(encoded_sizes), transform_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(transform_size, transform_size),
            nn.BatchNorm1d(transform_size)
        )
        self.central_encoder = nn.Sequential(
            nn.Linear(transform_size, central_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.central_decoder = nn.Sequential(
            nn.Linear(central_size, transform_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.pre_decode_transform = nn.Sequential(
            nn.Linear(transform_size, transform_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(transform_size, sum(encoded_sizes)),
            nn.BatchNorm1d(sum(encoded_sizes))
        )
        self.decoders = nn.ModuleList(
            [ModalityDecoder(encoded_size, input_size) for encoded_size, input_size in zip(encoded_sizes, input_sizes)])

        self.classifier = nn.Sequential(
            nn.Linear(central_size, 50),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(50, classifier_output_size),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        encoded = [encoder(input) for encoder, input in zip(self.encoders, inputs)]
        concatenated = torch.cat(encoded, dim=1)
        post_concat = self.post_concat_transform(concatenated)
        central_encoded = self.central_encoder(post_concat)

        central_decoded = self.central_decoder(central_encoded)
        pre_decode = self.pre_decode_transform(central_decoded)
        split_sizes = [encoder.encoder[0].out_features for encoder in self.encoders]
        split_decoded = torch.split(pre_decode, split_sizes, dim=1)
        decoded = [decoder(split) for decoder, split in zip(self.decoders, split_decoded)]

        classification_output = self.classifier(central_encoded)

        return *decoded, classification_output

    def encode(self, inputs):
        encoded = [self.encoders[i](inputs[i]) for i in range(len(inputs))]
        concatenated = torch.cat(encoded, dim=1)
        post_concat = self.post_concat_transform(concatenated)
        central_encoded = self.central_encoder(post_concat)
        return central_encoded


# Define the model parameters
input_sizes = [85, 10, 140, 13, 13, 46]
encoded_sizes = [30, 5, 40, 7, 7, 18]  # Adjust as needed
central_size = 50
transform_size = 80
classifier_output_size = 1

# Initialize the model
model = MultimodalAutoencoderWithClassifier(input_sizes, encoded_sizes, central_size, transform_size,
                                            classifier_output_size)

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare example input data
X1 = torch.randn(32, 85).to(device)  # Group 1: Program Data
X2 = torch.randn(32, 10).to(device)  # Group 2: College Data
X3 = torch.randn(32, 140).to(device)  # Group 3: Curriculum 1 Data
X4 = torch.randn(32, 13).to(device)  # Group 4: College 1 Data
X5 = torch.randn(32, 13).to(device)  # Group 5: Curric College Data
X6 = torch.randn(32, 46).to(device)  # Group 6: Other Features

# Forward pass through the model
model.eval()
with torch.no_grad():
    outputs = model([X1, X2, X3, X4, X5, X6])
    *reconstructions, classification_output = outputs

    # Check the shapes of the outputs
    # print("Reconstructions Shapes:", [recon.shape for recon in reconstructions])
    # print("Classification Output Shape:", classification_output.shape)

# Define the EarlyStopping class
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

model.to(device)

# Define loss functions
criterion_reconstruction = nn.MSELoss()
criterion_classification = nn.BCELoss()

# Define an optimizer with L2 regularization
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

# Example DataLoader setup
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

def train_model(model, train_loader, val_loader, criterion_reconstruction, criterion_classification, optimizer,
                num_epochs=100):
    training_losses = []
    validation_losses = []
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            *inputs, labels = batch
            inputs = [input.to(device) for input in inputs]
            labels = labels.to(device).float()

            optimizer.zero_grad()

            *reconstructions, classification_output = model(inputs)
            reconstruction_loss = sum(
                criterion_reconstruction(recon, input) for recon, input in zip(reconstructions, inputs))
            classification_loss = criterion_classification(classification_output.squeeze(), labels)
            l1_lambda = 0.0001
            l1_norm = sum(p.abs().sum() for p in model.parameters())

            total_loss = reconstruction_loss + classification_loss + l1_lambda * l1_norm

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()

        avg_train_loss = train_loss / len(train_loader)
        training_losses.append(avg_train_loss)

        # Validation loss calculation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                *inputs, labels = batch
                inputs = [input.to(device) for input in inputs]
                labels = labels.to(device).float()

                *reconstructions, classification_output = model(inputs)
                reconstruction_loss = sum(
                    criterion_reconstruction(recon, input) for recon, input in zip(reconstructions, inputs))
                classification_loss = criterion_classification(classification_output.squeeze(), labels)
                total_val_loss = reconstruction_loss + classification_loss

                val_loss += total_val_loss.item()

        avg_val_loss = val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)

        print(f'Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        # Early Stopping check
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return training_losses, validation_losses


# Call the train_model function and plot the losses
training_losses, validation_losses = train_model(model, train_loader, val_loader, criterion_reconstruction,
                                                 criterion_classification, optimizer, num_epochs=100)

# Plot the Training and Validation Losses
plt.figure(figsize=(10, 5))
plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.title('Training and Validation Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Model evaluation
def calculate_test_loss(model, test_loader, criterion_reconstruction, criterion_classification):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            *inputs, labels = batch
            inputs = [input.to(device) for input in inputs]
            labels = labels.to(device).float()
            *reconstructions, classification_output = model(inputs)
            reconstruction_loss = sum(criterion_reconstruction(recon, inputs[i]) for i, recon in enumerate(reconstructions))
            classification_loss = criterion_classification(classification_output.squeeze(), labels)
            test_loss += reconstruction_loss.item() + classification_loss.item()
    print(f'Average Test Loss: {test_loss / len(test_loader):.4f}')

calculate_test_loss(model, test_loader, criterion_reconstruction, criterion_classification)

# Model accuracy calculation
def evaluate_performance(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            *inputs, labels = batch
            inputs = [input.to(device) for input in inputs]
            labels = labels.to(device).float()
            outputs = model(inputs)
            classification_output = outputs[-1]
            predicted = (classification_output.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the model on the test data: {100 * correct / total:.2f}%')

evaluate_performance(model, test_loader)

#------------------------
# t-SNE visualization
#------------------------
def extract_features_and_visualize(model, data_loader):
    model.eval()
    features = []
    labels_list = []
    with torch.no_grad():
        for batch in data_loader:
            *inputs, labels = batch
            inputs = [input.to(device) for input in inputs]
            encoded_features = model.encode(inputs).cpu().numpy()
            features.append(encoded_features)
            labels_list.append(labels.cpu().numpy())

    features = np.concatenate(features)
    labels_list = np.concatenate(labels_list)

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(10, 6))
    for class_value in np.unique(labels_list):
        indices = labels_list == class_value
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=f'Class {class_value}')
    plt.xlabel('t-SNE feature 0')
    plt.ylabel('t-SNE feature 1')
    plt.title('t-SNE Visualization of Encoded Features')
    plt.legend()
    plt.show()

extract_features_and_visualize(model, test_loader)
