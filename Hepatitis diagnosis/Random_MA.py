import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/Users/divya/Desktop/Hepatitis copy.csv' 
data = pd.read_csv(file_path)

# Preprocess the data
data = data.drop(columns=['Unnamed: 0'])
data['Category'] = data['Category'].apply(lambda x: 0 if x == '0=Blood Donor' else 1)
data['Sex'] = data['Sex'].apply(lambda x: 0 if x == 'm' else 1)

# Handle missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Define features and target
X = data_imputed.drop(columns=['Category'])
y = data_imputed['Category']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Calculate confusion matrix
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# Calculate classification report
class_report_rf = classification_report(y_test, y_pred_rf)

# Calculate ROC curve and AUC
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_prob_rf)
auc_rf = roc_auc_score(y_test, y_prob_rf)

# Calculate specificity, sensitivity, and accuracy
tn_rf, fp_rf, fn_rf, tp_rf = conf_matrix_rf.ravel()
specificity_rf = tn_rf / (tn_rf + fp_rf)
sensitivity_rf = tp_rf / (tp_rf + fn_rf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# Print results
print("Confusion Matrix (Random Forest):")
print(conf_matrix_rf)
print("\nClassification Report (Random Forest):")
print(class_report_rf)
print(f"True Positives (TP): {tp_rf}")
print(f"True Negatives (TN): {tn_rf}")
print(f"False Positives (FP): {fp_rf}")
print(f"False Negatives (FN): {fn_rf}")
print(f"Sensitivity (Recall): {sensitivity_rf:.2f}")
print(f"Specificity: {specificity_rf:.2f}")
print(f"Accuracy: {accuracy_rf:.2f}")

# Plot the ROC curve
plt.figure()
plt.plot(fpr_rf, tpr_rf, color='blue', lw=2, label=f'ROC curve (area = {auc_rf:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot the sensitivity/specificity plot
plt.figure()
plt.plot(thresholds_rf, tpr_rf, label='Sensitivity (True Positive Rate)')
plt.plot(thresholds_rf, 1 - fpr_rf, label='Specificity (True Negative Rate)')
plt.xlabel('Threshold')
plt.ylabel('Rate')
plt.title('Sensitivity and Specificity vs. Threshold')
plt.legend(loc='best')
plt.show()
