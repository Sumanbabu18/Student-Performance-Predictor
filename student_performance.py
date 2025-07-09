# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load Dataset
data = pd.read_csv("StudentsPerformance.csv")
print("First 5 rows of the dataset:")
print(data.head())

# Step 3: Check for Nulls and Dataset Info
print("\nDataset Info:")
print(data.info())

# Step 4: Data Visualization (EDA)
plt.figure(figsize=(8, 5))
sns.boxplot(x='test preparation course', y='math score', data=data)
plt.title("Math Scores by Test Preparation Course")
plt.show()

# Step 5: Encode Categorical Data
label = LabelEncoder()
categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

for col in categorical_cols:
    data[col] = label.fit_transform(data[col])

# Step 6: Create New Feature (Average Score)
data['average_score'] = data[['math score', 'reading score', 'writing score']].mean(axis=1)

# Step 7: Define Features and Target
X = data.drop(['math score', 'reading score', 'writing score', 'average_score'], axis=1)
y = data['average_score']

# Step 8: Split Data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 10: Predict
y_pred = model.predict(X_test)

# Step 11: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Step 12: Plot Actual vs Predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='black')
plt.xlabel("Actual Average Scores")
plt.ylabel("Predicted Average Scores")
plt.title("Actual vs Predicted Student Scores")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.grid(True)
plt.show()

# Step 13: Save Model (Optional)
import joblib
joblib.dump(model, 'student_performance_predictor.pkl')
print("\nModel saved as student_performance_predictor.pkl")
