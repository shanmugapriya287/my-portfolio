import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, precision_score, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/Users/divya/Desktop/ DAPM Charts /Final_Preprocessed_data.csv'
data = pd.read_csv(file_path)

# Encode the categorical target variable
label_encoder = LabelEncoder()
data['Diagnosis'] = label_encoder.fit_transform(data['Diagnosis'])

# Split the data into features and target
X = data.drop('Diagnosis', axis=1)
y = data['Diagnosis']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM classifier
svm_classifier = SVC(probability=True)
svm_classifier.fit(X_train_scaled, y_train)

# Predict probabilities and classes on the test set and training set
y_pred_proba_test = svm_classifier.predict_proba(X_test_scaled)[:,1]
y_pred_test = svm_classifier.predict(X_test_scaled)

y_pred_proba_train = svm_classifier.predict_proba(X_train_scaled)[:,1]
y_pred_train = svm_classifier.predict(X_train_scaled)

# Compute ROC curve and AUC for both test and train data
fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_proba_test)
roc_auc_test = auc(fpr_test, tpr_test)

fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_proba_train)
roc_auc_train = auc(fpr_train, tpr_train)

# Compute Precision, F1 Score, and Accuracy for both test and train data
precision_test = precision_score(y_test, y_pred_test)
f1_score_test = f1_score(y_test, y_pred_test)
accuracy_test = accuracy_score(y_test, y_pred_test)

precision_train = precision_score(y_train, y_pred_train)
f1_score_train = f1_score(y_train, y_pred_train)
accuracy_train = accuracy_score(y_train, y_pred_train)

# Function to plot confusion matrix
def plot_confusion_matrix(conf_matrix, title='Confusion Matrix'):
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Compute confusion matrices for both test and train data
conf_matrix_test = confusion_matrix(y_test, y_pred_test)
conf_matrix_train = confusion_matrix(y_train, y_pred_train)

# Plotting the ROC curve
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, label='Test ROC curve (area = %0.2f)' % roc_auc_test)
plt.plot(fpr_train, tpr_train, color='blue', lw=2, label='Train ROC curve (area = %0.2f)' % roc_auc_train)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Printing evaluation metrics
print("Test Data Metrics:")
print(f"Precision: {precision_test:.2f}, F1 Score: {f1_score_test:.2f}, Accuracy: {accuracy_test:.2f}")

print("\nTrain Data Metrics:")
print(f"Precision: {precision_train:.2f}, F1 Score: {f1_score_train:.2f}, Accuracy: {accuracy_train:.2f}")

# Show ROC curve
plt.show()

# Plot confusion matrices
print("Test Data Confusion Matrix:")
plot_confusion_matrix(conf_matrix_test, title='Confusion Matrix for Test Data')

print("\nTrain Data Confusion Matrix:")
plot_confusion_matrix(conf_matrix_train, title='Confusion Matrix for Train Data')
