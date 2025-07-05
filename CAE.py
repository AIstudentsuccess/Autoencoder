# Libraries 
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.utils.class_weight import compute_class_weight 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# Load the main dataset
data = pd.read_csv("cleaned_sample_data.csv", low_memory=False)

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

# Convert the entire DataFrame to tensors 
x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
x_val_tensor = torch.tensor(x_val.values, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Convert to numpy arrays
x_train_array = x_train.values
x_val_array = x_val.values
x_test_array = x_test.values

# Reshape for 1D CNN input
x_train_reshaped = np.expand_dims(x_train_array, axis=1)
x_val_reshaped = np.expand_dims(x_val_array, axis=1)
x_test_reshaped = np.expand_dims(x_test_array, axis=1)

# Convert the entire DataFrame to tensors 
x_train_tensor = torch.tensor(x_train_reshaped, dtype=torch.float32)
x_val_tensor = torch.tensor(x_val_reshaped, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test_reshaped, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1) 
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)  
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)  

# Create TensorDatasets
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#------------------------------------------------------------------------------------------
# Convolutional Autoencoder Model
#------------------------------------------------------------------------------------------

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=20, stride=1, padding=1), # 164 sample data, 180 real data
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(8),  
            nn.MaxPool1d(2, stride=1),  # 163 sample data, 179 real data
            nn.Dropout(0.1),  
            
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=20, stride=1, padding=1), # 146 sample data, 162 real data
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(16),  
            nn.MaxPool1d(2, stride=1),  # 145 sample data, 161 real data
            nn.Dropout(0.1),  
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=20, stride=1, padding=1), # 128 sample data, 144 real data
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(32),  
            nn.MaxPool1d(2, stride=1),  # 127 sample data, 143 real data
            nn.Dropout(0.1),  
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=20, stride=1, padding=1), # 110 sample data, 126 real data
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(64),  
            nn.MaxPool1d(2, stride=1),  # 109 sample data, 125 real data
            nn.Dropout(0.1), 
            
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=20, stride=1, padding=1), # 92 sample data, 108 real data
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(128),  
            nn.MaxPool1d(2, stride=1),  # 91 sample data, 107 real data
            nn.Dropout(0.1),  
            
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=20, stride=1, padding=1), # 74 sample data, 90 real data
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(256),  
            nn.MaxPool1d(2, stride=1),  # 73 sample data, 89 real data
            nn.Dropout(0.1), 
            
            nn.Flatten(),  
            nn.Linear(256 * 73, 141),  
            nn.LeakyReLU(negative_slope=0.01)
        )
        
        # Decoder 
        self.decoder = nn.Sequential(
            nn.Linear(141, 256 * 73),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Unflatten(1, (256, 73)),
            
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=21, stride=1, padding=1), # 91 sample data, 107 real data
            nn.LeakyReLU(negative_slope=0.01),
            #nn.BatchNorm1d(128),  
            #nn.Dropout(0.1),
            
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=21, stride=1, padding=1), # 109 sample data, 125 real data
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(64),  
            nn.Dropout(0.1),
            
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=21, stride=1, padding=1), # 127 sample data, 143 real data
            nn.LeakyReLU(negative_slope=0.01),
            #nn.BatchNorm1d(32), 
            #nn.Dropout(0.1),
            
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=21, stride=1, padding=1), # 145 sample data, 161 real data
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm1d(16), 
            nn.Dropout(0.1),
            
            nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=21, stride=1, padding=1), # 163 sample data, 179 real data
            nn.LeakyReLU(negative_slope=0.01),
            #nn.BatchNorm1d(8), 
            #nn.Dropout(0.1),
            
            nn.ConvTranspose1d(in_channels=8, out_channels=1, kernel_size=21, stride=1, padding=1) # 181 sample data, 197 real data
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        with torch.no_grad():  
            encoded = self.encoder(x)
        return encoded

# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)    
    
autoencoder = ConvAutoencoder().to(device)
print(autoencoder)    

#------------------------------------------------------------------------------------------
# Convolutional Autoencoder Training
#------------------------------------------------------------------------------------------

criterion = nn.MSELoss()
criterion = criterion.to(device)
optimizer = optim.Adam(autoencoder.parameters(), lr=0.0000001, weight_decay=0.00001)

lambda_l1 = 0.000005 

best_val_loss = float('inf')
patience = 10
wait = 0
early_stop = False

train_losses = []
val_losses = []

num_epochs = 300
for epoch in range(num_epochs):
    autoencoder.train()
    train_loss = 0.0
    for data in train_loader:
        inputs, _ = data
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs = autoencoder(inputs)
        mse_loss = criterion(outputs, inputs)
        l1_loss = sum(torch.sum(torch.abs(param)) for param in autoencoder.parameters())
        loss = mse_loss + lambda_l1 * l1_loss  
        loss.backward()
        optimizer.step()
        train_loss += mse_loss.item() * inputs.size(0)  
    train_loss = train_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    autoencoder.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            inputs, _ = data
            inputs = inputs.to(device)
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            val_loss += loss.item() * inputs.size(0)
    val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
        torch.save(autoencoder.state_dict(), 'best_model.pth')
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered")
            early_stop = True
            break

if early_stop:
    print("Training stopped early")

# Saving the Model Weights
torch.save(autoencoder.state_dict(), 'autoencoder_weights.pth')

#------------------------------------------------------------------------------------------
# Plot Training And Validation Loss
#------------------------------------------------------------------------------------------

# Set figure parameters
params = {
    'axes.labelsize': 12,
    'font.size': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': False,
    'figure.figsize': [4.5, 4.5]
}
rcParams.update(params)
fig, ax = plt.subplots(figsize=(4.5, 4.5), constrained_layout=True) 
ax.plot(train_losses, linewidth=2, color='#B22400', label='Training Loss')  
ax.plot(val_losses, linewidth=2, linestyle='--', color='#006BB2', label='Validation Loss') 
ax.set_title('Training and Validation Loss')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.margins(x=0.03, y=0.03)  
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
legend = ax.legend(loc='upper right')  
frame = legend.get_frame()
frame.set_facecolor('0.9')  
frame.set_edgecolor('0.9')  
inset_ax = inset_axes(ax, width="40%", height="40%", loc='center right')  
inset_ax.plot(train_losses[:20], linewidth=2, color='#B22400', label='Training Loss')  
inset_ax.plot(val_losses[:20], linewidth=2, linestyle='--', color='#006BB2', label='Validation Loss')
inset_ax.set_title('Early Training Behavior', fontsize=8)
inset_ax.set_xlabel('', fontsize=8)
inset_ax.set_ylabel('', fontsize=8)
inset_ax.tick_params(labelsize=8)
inset_ax.grid(True, linestyle='--', linewidth=0.5)
plt.show()

#------------------------------------------------------------------------------------------
# Test Loss
#------------------------------------------------------------------------------------------

autoencoder.eval()  
test_loss = 0.0
with torch.no_grad(): 
    for data in test_loader:
        inputs, _ = data
        inputs = inputs.to(device)
        outputs = autoencoder(inputs)
        loss = criterion(outputs, inputs)
        test_loss += loss.item() * inputs.size(0)
test_loss = test_loss / len(test_loader.dataset)

print(f'Test Loss: {test_loss:.4f}')

#------------------------------------------------------------------------------------------
# 3D t-SNE Visualization of Encoded Features
#------------------------------------------------------------------------------------------

encoded_features_list = []
labels_list = []

autoencoder.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        encoded_features = autoencoder.encode(inputs).cpu()
        encoded_features_list.append(encoded_features)
        labels_list.append(labels)

encoded_features = torch.cat(encoded_features_list).numpy()
labels = torch.cat(labels_list).numpy()
tsne = TSNE(n_components=3, perplexity=30, n_iter=1000, random_state=123)
encoded_features_3d = tsne.fit_transform(encoded_features)
fig = plt.figure(figsize=(4.5, 4.5))  
ax = fig.add_subplot(111, projection='3d')
ax.scatter(encoded_features_3d[:, 0], encoded_features_3d[:, 1], encoded_features_3d[:, 2],
           c=labels, cmap='viridis', alpha=0.5)

ax.set_title('3D t-SNE Visualization of Encoded Features', fontsize=10)
ax.set_xlabel('t-SNE feature 1', fontsize=10)
ax.set_ylabel('t-SNE feature 2', fontsize=10)
ax.set_zlabel('t-SNE feature 3', fontsize=10)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.show()

#------------------------------------------------------------------------------------------
# Random Forest Model On Embeddings
#------------------------------------------------------------------------------------------

autoencoder = ConvAutoencoder()
autoencoder.load_state_dict(torch.load('autoencoder_weights.pth'))
autoencoder.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_embeddings(loader, autoencoder, device):
    autoencoder.to(device)
    embeddings = []
    targets = []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            encoded_features = autoencoder.encode(data)
            embeddings.append(encoded_features.cpu().numpy())
            targets.append(target.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    targets = np.concatenate(targets, axis=0)
    return embeddings, targets

# Extract embeddings for the training, validation, and test sets
train_embeddings, train_labels = extract_embeddings(train_loader, autoencoder, device)
val_embeddings, val_labels = extract_embeddings(val_loader, autoencoder, device)
test_embeddings, test_labels = extract_embeddings(test_loader, autoencoder, device)

# Flatten the labels
train_labels = np.ravel(train_labels)
val_labels = np.ravel(val_labels)
test_labels = np.ravel(test_labels)

# Prepare class weights due to imbalance
y_np = y.to_numpy()  
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_np), y=y_np)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, class_weight=class_weights_dict, random_state=123)

# Perform five-fold cross-validation to estimate model performance
cv_scores = cross_val_score(rf_classifier, train_embeddings, train_labels, cv=5, scoring='accuracy')
print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f}")
print(f"Standard Deviation of CV Accuracy: {np.std(cv_scores):.4f}")

# Evaluate on the validation set before final training
rf_classifier.fit(train_embeddings, train_labels)
val_predictions = rf_classifier.predict(val_embeddings)
print(f"Validation Accuracy: {accuracy_score(val_labels, val_predictions):.4f}")
print("Validation Set Classification Report:\n", classification_report(val_labels, val_predictions))

# Combine training and validation sets for final model training
combined_embeddings = np.concatenate((train_embeddings, val_embeddings), axis=0)
combined_labels = np.concatenate((train_labels, val_labels), axis=0)
rf_classifier.fit(combined_embeddings, combined_labels)

# Make predictions on the test set for final evaluation
test_predictions = rf_classifier.predict(test_embeddings)
print("Test Set Random Forest Classifier report: \n", classification_report(test_labels, test_predictions))

#------------------------------------------------------------------------------------------
# ROC Curve On Embeddings
#------------------------------------------------------------------------------------------

# Set figure parameters 
params = {
    'axes.labelsize': 8,
    'font.size': 8,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': False,
    'figure.figsize': [4.5, 4.5]
}
rcParams.update(params)

test_probabilities = rf_classifier.predict_proba(test_embeddings)[:, 1]
fpr, tpr, thresholds = roc_curve(test_labels, test_probabilities)
roc_auc = auc(fpr, tpr)

# Plotting ROC Curve
plt.figure(figsize=(4.5, 4.5))  
plt.plot(fpr, tpr, color='#B22400', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve on Embeddings')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  
plt.legend(loc="lower right")
plt.show()

#------------------------------------------------------------------------------------------
# Confusion Matrix Test Set On Embeddings
#------------------------------------------------------------------------------------------

# Compute confusion matrix
cm = confusion_matrix(test_labels, test_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Greys, values_format='.0f')  # Adjust format to integer display

for text in disp.text_.ravel():
    bbox_patch = text.get_bbox_patch()
    if bbox_patch is not None:
        background_color = bbox_patch.get_facecolor()
        color_intensity = np.mean(background_color[:3])  
        text_color = 'white' if color_intensity < 0.5 else 'black'
        text.set_color(text_color)
    text.set_fontsize(12)  
plt.title('Test Set on Embeddings')
plt.grid(False) 
plt.show()

#------------------------------------------------------------------------------------------
# Confusion Matrix Validation Set On Embeddings
#------------------------------------------------------------------------------------------

# Compute confusion matrix
cm = confusion_matrix(val_labels, val_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Greys, values_format='.0f')  # Adjust format to integer display

for text in disp.text_.ravel():
    bbox_patch = text.get_bbox_patch()
    if bbox_patch is not None:
        background_color = bbox_patch.get_facecolor()
        color_intensity = np.mean(background_color[:3])  
        text_color = 'white' if color_intensity < 0.5 else 'black'
        text.set_color(text_color)
    text.set_fontsize(12)  
plt.title('Validation set on Embeddings')
plt.grid(False)  
plt.show()
