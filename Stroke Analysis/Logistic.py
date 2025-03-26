import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, roc_curve, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/Users/divya/Desktop/ DAPM Charts /Final_Preprocessed_data.csv'  
data = pd.read_csv(file_path)

# Encoding the target variable
label_encoder = LabelEncoder()
data['Diagnosis'] = label_encoder.fit_transform(data['Diagnosis'])

# Splitting the dataset into training and testing sets
X = data.drop('Diagnosis', axis=1)
y = data['Diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeling - Logistic Regression
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train, y_train)

# Validation on Test Set
y_test_pred = logistic_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_roc_auc = roc_auc_score(y_test, y_test_pred)

# Validation on Training Set
y_train_pred = logistic_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)
train_roc_auc = roc_auc_score(y_train, y_train_pred)


# ROC Curve Data
fpr_test, tpr_test, _ = roc_curve(y_test, logistic_model.predict_proba(X_test)[:, 1])
roc_auc_test = roc_auc_score(y_test, y_test_pred)

fpr_train, tpr_train, _ = roc_curve(y_train, logistic_model.predict_proba(X_train)[:, 1])
roc_auc_train = roc_auc_score(y_train, y_train_pred)

# Plotting ROC Curves
plt.figure()
plt.plot(fpr_test, tpr_test, label=f'Test Set ROC curve (area = {roc_auc_test:.2f})')
plt.plot(fpr_train, tpr_train, label=f'Training Set ROC curve (area = {roc_auc_train:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Function to plot Confusion Matrix
def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(title)
    plt.show()

# Plotting Confusion Matrix for Test Set
plot_conf_matrix(y_test, y_test_pred, 'Confusion Matrix (Test Set)')

# Plotting Confusion Matrix for Training Set
plot_conf_matrix(y_train, y_train_pred, 'Confusion Matrix (Training Set)')

# Printing Results
print("Training Set Metrics:")
print(f"Accuracy: {train_accuracy}")
print(f"Precision: {train_precision}")
print(f"Recall: {train_recall}")
print(f"ROC-AUC Score: {train_roc_auc}")

print("\nTest Set Metrics:")
print(f"Accuracy: {test_accuracy}")
print(f"Precision: {test_precision}")
print(f"Recall: {test_recall}")
print(f"ROC-AUC Score: {test_roc_auc}")
