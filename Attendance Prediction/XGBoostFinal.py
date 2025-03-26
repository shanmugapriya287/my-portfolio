import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

print("Start processing:")

# Load the data
file_path = '/Users/divya/Downloads/for_python.xlsx'  
data = pd.read_excel(file_path)

# Preprocess the data
# Convert datetime columns to numerical values
data['Attendance Detail Date'] = pd.to_datetime(data['Attendance Detail Date'])
data['First Entry Date'] = pd.to_datetime(data['First Entry Date'])
data['Days Since First Entry'] = (data['Attendance Detail Date'] - data['First Entry Date']).dt.days

# Drop original datetime columns if they are not needed
data = data.drop(columns=['Attendance Detail Date', 'First Entry Date', 'Days Since first entry'])

# Encode categorical variables
categorical_cols = ['Entry Time Category', 'Attendance Detail Weekday', 'Contacts Detail Gender', 'Contacts Detail Price Level', 'Membership Level']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Store the label encoder for potential inverse transformation

# Features to use in prediction
X = data[['Contacts Detail Age', 'Attendance Detail Weekday', 'Membership Level', 'Contacts Detail Price Level', 'Days Since First Entry']]

# List of activities to predict
activities = ['Gym Accessed', 'Pool Accessed', 'Classes', 'Other Activities', 'Alternative Sessions']

# Initialize lists to collect overall scores and predicted vs actual data
overall_r2_scores = []
overall_mse_scores = []
overall_rmse_scores = []
combined_results_df = pd.DataFrame()

for activity in activities:
    # Target variable (binary encoded)
    y = data[activity]
    
    # Split the data into training and testing sets for regression
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    # Initialize the XGBoost regressor
    model = XGBRegressor(random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the regression model
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    # Collect overall scores
    overall_r2_scores.append(r2)
    overall_mse_scores.append(mse)
    overall_rmse_scores.append(rmse)
    
    # Combine predicted vs actual values into a single DataFrame
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Activity': activity})
    combined_results_df = pd.concat([combined_results_df, results_df])

    # Feature importance with rounded percentages
    feature_importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    })
    feature_importance_df['Importance_Percent'] = (100 * feature_importance_df['Importance'] / feature_importance_df['Importance'].sum()).round(0).astype(int)
    
    # Display feature importances with whole number percentages
    print(f"Feature Importances for predicting {activity}:")
    print(feature_importance_df[['Feature', 'Importance_Percent']].sort_values(by='Importance_Percent', ascending=False))
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance_Percent', y='Feature', data=feature_importance_df.sort_values(by='Importance_Percent', ascending=False))
    plt.title(f'Feature Importances with Percentages for {activity}')
    plt.xlabel('Importance Percentage')
    plt.ylabel('Feature')
    plt.show()

# Display only the first 5 values for all activities together
print("\nFirst 5 Predicted vs Actual values for all activities combined:\n")
print(combined_results_df.head(5))

# Print the overall scores
print("\nOverall Scores for All Activities:")
print(f"Average R² Value: {sum(overall_r2_scores) / len(overall_r2_scores)}")
print(f"Average Mean Squared Error: {sum(overall_mse_scores) / len(overall_mse_scores)}")
print(f"Average Root Mean Squared Error: {sum(overall_rmse_scores) / len(overall_rmse_scores)}")

# Identify the top 5 attendees based on the 'Total' column
top_5_attendees = data.nlargest(5, 'Total')

# Reverse the encoding for categorical columns
top_5_attendees['Membership Level'] = label_encoders['Membership Level'].inverse_transform(top_5_attendees['Membership Level'])
top_5_attendees['Contacts Detail Price Level'] = label_encoders['Contacts Detail Price Level'].inverse_transform(top_5_attendees['Contacts Detail Price Level'])
top_5_attendees['Contacts Detail Gender'] = label_encoders['Contacts Detail Gender'].inverse_transform(top_5_attendees['Contacts Detail Gender'])

# Display the top 5 attendees with their unique ID, price level, gender, age, membership level, and activities accessed
print("\nTop 5 Attendees:")
print(top_5_attendees[['Unique Key', 'Contacts Detail Price Level', 'Contacts Detail Gender', 'Contacts Detail Age', 
                       'Membership Level', 'Gym Accessed', 'Pool Accessed', 'Classes', 'Other Activities', 
                       'Alternative Sessions']])
