import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load data from a CSV downloaded from Kaggle
df = pd.read_csv("student_scores.csv")

# Example columns: Hours, Scores
X = df[["Study_Hours"]]
y = df["Scores"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)

n = len(y_test)
p = X_test.shape[1]
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])
print("R-squared:", r2)
print("Adjusted R-squared:", adjusted_r2)

rs = float(input("enter the number of hours you studied: "))
predicted_marks = model.predict([[rs]])
print(f" based on the number of hours you studied: {rs}, your marks may be {predicted_marks[0]}")