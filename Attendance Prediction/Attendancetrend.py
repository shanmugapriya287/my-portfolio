import pandas as pd
import matplotlib.pyplot as plt

print("start processing:")
# Load the data
file_path = '/Users/divya/Documents/Copy of final_for_analysiss (1).xlsx'
data = pd.read_excel(file_path)


# Converting the 'Attendance Detail Date' to a datetime format for easier manipulation
data['Attendance Detail Date'] = pd.to_datetime(data['Attendance Detail Date'])

# 1. Calculate overall attendance trends by aggregating attendance by date
attendance_trend = data.groupby('Attendance Detail Date')['Total Attendance '].sum().reset_index()

# 2. Analyze attendance based on weekdays
attendance_by_weekday = data.groupby('Attendance Detail Weekday')['Total Attendance '].sum().reset_index()

# 3. Correlate attendance with gender
attendance_by_gender = data.groupby('Contacts Detail Gender')['Total Attendance '].sum().reset_index()

# Plotting all three charts in a single figure
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# 1. Overall Attendance Trend Over Time
axs[0].plot(attendance_trend['Attendance Detail Date'], attendance_trend['Total Attendance '], marker='o', linestyle='-', color='blue')
axs[0].set_title('Overall Attendance Trend Over Time')
axs[0].set_xlabel('Date')
axs[0].set_ylabel('Total Attendance')
axs[0].grid(True)
axs[0].tick_params(axis='x', rotation=45)

# 2. Total Attendance by Weekday
axs[1].bar(attendance_by_weekday['Attendance Detail Weekday'], attendance_by_weekday['Total Attendance '], color='skyblue')
axs[1].set_title('Total Attendance by Weekday')
axs[1].set_xlabel('Weekday')
axs[1].set_ylabel('Total Attendance')
axs[1].grid(axis='y')

# 3. Total Attendance by Gender
axs[2].bar(attendance_by_gender['Contacts Detail Gender'], attendance_by_gender['Total Attendance '], color='green')
axs[2].set_title('Total Attendance by Gender')
axs[2].set_xlabel('Gender')
axs[2].set_ylabel('Total Attendance')
axs[2].grid(axis='y')

# Adjust layout for better spacing
plt.tight_layout()

plt.show()