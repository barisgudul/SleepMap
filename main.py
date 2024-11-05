# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load and clean dataset
# Removing rows with missing values for simplicity in this analysis
dataset = pd.read_csv("msleep.csv")
dataset = dataset.dropna()

# Feature and target variable selection
# X: Selecting relevant features for modeling
# y: Target variable (assuming it is at index 5 in the dataset)
X = dataset.iloc[:, 6:11]
y = dataset.iloc[:, 5]

# Split data into training and test sets
# 80% for training and 20% for testing to evaluate model performance
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regressor.predict(X_test)

# Calculate prediction error as the absolute difference between predictions and actual values
error = np.abs(y_pred - y_test.values)

# Organize data for error analysis by animal species
data = pd.DataFrame({
    "name": dataset["name"][y_test.index],
    "error": error
})

# Calculate the mean error for each species
error_by_species = data.groupby("name")["error"].mean()

# Plot the error distribution by species
plt.figure(figsize=(15, 8))
sns.heatmap(error_by_species.to_frame(), annot=True, cmap="viridis", cbar_kws={'label': 'Mean Absolute Error'})
plt.title("Error Distribution by Animal Species")
plt.xlabel("Animal Species")
plt.ylabel("Mean Absolute Error")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()  # Ensures plot elements are nicely fit into the figure
plt.show()
