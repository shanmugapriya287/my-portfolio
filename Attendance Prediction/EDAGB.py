import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.colors as mcolors

print("Start processing:")

# Load the data
file_path = '/Users/divya/Downloads/for_python.xlsx'  
data = pd.read_excel(file_path)

# EDA: Initial Data Exploration
print("Initial Data Exploration:")
print(data.info())
print("\nMissing values in each column:\n", data.isnull().sum())
print("\nSummary statistics:\n", data.describe())

# EDA: Visualize distributions of numerical features
plt.figure(figsize=(12, 8))
sns.histplot(data['Contacts Detail Age'], kde=True, bins=30)
plt.title('Distribution of Contacts Detail Age')
plt.show()

# Ensure 'Membership Level' is treated as a categorical variable
data['Membership Level'] = data['Membership Level'].astype('category')

# Visualize categorical features
plt.figure(figsize=(12, 8))
sns.countplot(x=data['Membership Level'])
plt.title('Count of Membership Level')
plt.show()

# Remove outliers using Z-score method for numerical features
z_scores = np.abs(stats.zscore(data[['Contacts Detail Age', 'Days Since First Entry']]))
data = data[(z_scores < 3).all(axis=1)]

# Recalculate descriptive statistics after removing outliers
print("\nSummary statistics after removing outliers:\n", data.describe())

# Preprocess the data
# Convert datetime columns to numerical values
data['Attendance Detail Date'] = pd.to_datetime(data['Attendance Detail Date'])
data['First Entry Date'] = pd.to_datetime(data['First Entry Date'])
data['Days Since First Entry'] = (data['Attendance Detail Date'] - data['First Entry Date']).dt.days

# Drop original datetime columns if they are not needed
data = data.drop(columns=['Attendance Detail Date', 'First Entry Date'])

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
    
    # Initialize the Gradient Boosting Regressor
    model = GradientBoostingRegressor(random_state=42)
    
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
    
    # Create a color palette from light to dark based on the #D3AE11 color
    base_color = '#D3AE11'
    num_colors = len(feature_importance_df)
    palette = sns.light_palette(mcolors.hex2color(base_color), n_colors=num_colors, reverse=True)
    
    # Plot feature importances with custom color palette and bold axis headers
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance_Percent', y='Feature', data=feature_importance_df.sort_values(by='Importance_Percent', ascending=False), palette=palette)
    plt.title(f'Feature Importances with Percentages for {activity}', fontsize=16, fontweight='bold')
    
    # Bold axis labels
    plt.xlabel('Importance Percentage', fontsize=14, fontweight='bold')
    plt.ylabel('Feature', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
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
