import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# Load dataset
file_path = '/Users/divya/Desktop/Python projects/Stroke_preprocessed.csv'  
data = pd.read_csv(file_path)


# Preparing data for classification
X = data.drop(['Diagnosis'], axis=1)  # Features
y = data['Diagnosis']  # Target variable

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predictions
y_pred = rf_classifier.predict(X_test)

# Evaluation
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Output the classification report and confusion matrix
print(classification_rep)
print(conf_matrix)
