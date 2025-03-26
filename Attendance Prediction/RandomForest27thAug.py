import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

print("Start processing with Random Forest Regressor:")

# Load the data
file_path = '/Users/divya/Downloads/for_python.xlsx'  
data = pd.read_excel(file_path)

# Convert datetime columns to numerical values
data['Attendance Detail Date'] = pd.to_datetime(data['Attendance Detail Date'])
data['First Entry Date'] = pd.to_datetime(data['First Entry Date'])

# Calculate 'Frequency of Entry' as the count of each Unique Key
data['Frequency of Entry'] = data.groupby('Unique Key')['Unique Key'].transform('count')

# Extract month name from 'First Entry Date' and encode it
data['First Entry Month'] = data['First Entry Date'].dt.strftime('%B')  # Convert to month name (e.g., "October")

# Drop original datetime columns if they are not needed
data = data.drop(columns=['Attendance Detail Date', 'First Entry Date'])

# Encode categorical variables
categorical_cols = ['Entry Time Category', 'Attendance Detail Weekday', 'Contacts Detail Gender', 'Contacts Detail Price Level', 'Membership Level', 'First Entry Month']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Store the label encoder for potential inverse transformation

# Features used for prediction 
X = data[['Contacts Detail Age', 'Contacts Detail Gender', 'Attendance Detail Weekday', 'Membership Level', 'Contacts Detail Price Level']]


# Modify feature names to include line breaks for better visibility
X.columns = [
    'Contacts Detail\nAge',
    'Attendance Detail\nWeekday',
    'Membership\nLevel',
    'Contacts Detail\nPrice Level',
    'Contacts Detail\nGender'
]


# List of activities to predict
activities = ['Gym Accessed', 'Pool Accessed', 'Classes', 'Squash', 'Other Activities', 'Alternative Sessions']

# Initialize lists to collect overall scores and predicted vs actual data
overall_r2_scores_rf = []
overall_mse_scores_rf = []
overall_rmse_scores_rf = []
combined_results_df_rf = pd.DataFrame()

models_rf = {}

for activity in activities:
    # Target variable (binary encoded)
    y = data[activity]
    
    # Split the data into training and testing sets for regression
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Initialize the Random Forest Regressor
    model_rf = RandomForestRegressor(random_state=42)
    
    # Train the model
    model_rf.fit(X_train, y_train)
    
    # Store the model for later use in predictions
    models_rf[activity] = model_rf
    
    # Make predictions
    y_pred_rf = model_rf.predict(X_test)
    
    # Evaluate the regression model
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
    r2_rf = r2_score(y_test, y_pred_rf)
    
    # Collect overall scores
    overall_r2_scores_rf.append(r2_rf)
    overall_mse_scores_rf.append(mse_rf)
    overall_rmse_scores_rf.append(rmse_rf)
    
    # Combine predicted vs actual values into a single DataFrame
    results_df_rf = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_rf, 'Activity': activity})
    combined_results_df_rf = pd.concat([combined_results_df_rf, results_df_rf])

    # Plot feature importances for each activity
    feature_importances_rf = model_rf.feature_importances_
    feature_importance_df_rf = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances_rf
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df_rf, palette='viridis')
    plt.title(f'Feature Importance for {activity} - Random Forest', fontsize=17, fontweight='bold')
    plt.xlabel('Importance', fontsize=14, fontweight='bold')
    plt.ylabel('Feature', fontsize=14, fontweight='bold')
    plt.show()

# Display only the first 5 values for all activities together
print("\nFirst 5 Predicted vs Actual values for all activities combined (Random Forest):\n")
print(combined_results_df_rf.head(5))

# Print the overall scores
print("\nOverall Scores for All Activities (Random Forest):")
print(f"Average R² Value: {sum(overall_r2_scores_rf) / len(overall_r2_scores_rf)}")
print(f"Average Mean Squared Error: {sum(overall_mse_scores_rf) / len(overall_mse_scores_rf)}")
print(f"Average Root Mean Squared Error: {sum(overall_rmse_scores_rf) / len(overall_rmse_scores_rf)}")

# Plotting the predicted vs actual values
plt.figure(figsize=(14, 10))
for activity in activities:
    plt.scatter(combined_results_df_rf[combined_results_df_rf['Activity'] == activity]['Actual'], combined_results_df_rf[combined_results_df_rf['Activity'] == activity]['Predicted'], label=activity, alpha=0.5)
plt.plot([combined_results_df_rf['Actual'].min(), combined_results_df_rf['Actual'].max()],
         [combined_results_df_rf['Actual'].min(), combined_results_df_rf['Actual'].max()], 'r--', lw=2)
plt.title('Predicted vs Actual Values (Random Forest)', fontsize=16, fontweight='bold')
plt.xlabel('Actual Values', fontsize=14, fontweight='bold')
plt.ylabel('Predicted Values', fontsize=14, fontweight='bold')
plt.legend(title='Activity', fontsize=14)
plt.show()

# Identify the top 5 attendees based on the 'Total' column
top_5_attendees = data.nlargest(5, 'Total')

# Reverse the encoding for categorical columns
top_5_attendees['Membership Level'] = label_encoders['Membership Level'].inverse_transform(top_5_attendees['Membership Level'])
top_5_attendees['Contacts Detail Price Level'] = label_encoders['Contacts Detail Price Level'].inverse_transform(top_5_attendees['Contacts Detail Price Level'])
top_5_attendees['Contacts Detail Gender'] = label_encoders['Contacts Detail Gender'].inverse_transform(top_5_attendees['Contacts Detail Gender'])

# Display the top 5 attendees with their unique ID, price level, gender, age, membership level, and activities accessed
print("\nTop 5 Attendees:")
print(top_5_attendees[['Unique Key', 'Contacts Detail Price Level', 'Contacts Detail Gender', 'Contacts Detail Age', 
                       'Membership Level', 'Gym Accessed', 'Pool Accessed', 'Classes', 'Squash', 'Other Activities', 
                       'Alternative Sessions']])

# Function to get member details and predict activities for a given unique key
def get_member_details_and_predict(unique_key):
    # Search for the row with the given Unique Key
    member_data = data[data['Unique Key'] == unique_key]
    
    # Check if the member exists
    if member_data.empty:
        print(f"No data found for Unique Key: {unique_key}")
        return
    
    # Print the found data for debugging
    print("Selected member data:\n", member_data[['Unique Key', 'Frequency of Entry']])

    # Decode categorical columns to display
    member_data_display = member_data.copy()
    member_data_display['Contacts Detail Gender'] = label_encoders['Contacts Detail Gender'].inverse_transform(member_data['Contacts Detail Gender'])
    member_data_display['Membership Level'] = label_encoders['Membership Level'].inverse_transform(member_data['Membership Level'])
    member_data_display['Contacts Detail Price Level'] = label_encoders['Contacts Detail Price Level'].inverse_transform(member_data['Contacts Detail Price Level'])
    member_data_display['Entry Time Category'] = label_encoders['Entry Time Category'].inverse_transform(member_data['Entry Time Category'])
    member_data_display['Attendance Detail Weekday'] = label_encoders['Attendance Detail Weekday'].inverse_transform(member_data['Attendance Detail Weekday'])
    member_data_display['First Entry Month'] = label_encoders['First Entry Month'].inverse_transform(member_data['First Entry Month'])

    # Print the details
    print("\nMember Details:")
    print(f"Unique Key: {unique_key}")
    print(f"Gender: {member_data_display['Contacts Detail Gender'].values[0]}")
    print(f"Age: {member_data_display['Contacts Detail Age'].values[0]}")
    print(f"Membership Level: {member_data_display['Membership Level'].values[0]}")
    print(f"Price Level: {member_data_display['Contacts Detail Price Level'].values[0]}")
    print(f"Entry Time Category: {member_data_display['Entry Time Category'].values[0]}")
    print(f"Frequency of Entry: {member_data_display['Frequency of Entry'].values[0]}")
    print(f"Attendance Detail Weekday: {member_data_display['Attendance Detail Weekday'].values[0]}")
    print(f"First Entry Month: {member_data_display['First Entry Month'].values[0]}")
    
    # Predict activities
    input_data = member_data[X.columns]
    predictions = {}
    for activity, model in models_rf.items():
        predictions[activity] = model.predict(input_data)[0]
    
    print("\nPredicted Activities:")
    for activity, prediction in predictions.items():
        print(f"{activity}: {prediction}")

unique_id = int(input("Enter the Unique Key: "))
get_member_details_and_predict(unique_id)
