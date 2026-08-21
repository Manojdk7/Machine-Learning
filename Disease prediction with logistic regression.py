import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create a sample dataset for disease prediction

data = {
    "age": [45, 50, 60, 35, 70, 42, 55, 65, 40, 58],
    "blood_pressure": [120, 130, 140, 110, 165, 118, 135, 150, 112, 142],
    "cholesterol": [180, 200, 210, 170, 240, 175, 205, 230, 172, 220],
    "sugar_level": [90, 100, 120, 85, 180, 88, 110, 170, 86, 130],
    "disease": [0, 0, 1, 0, 1, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)
print(df)

#Prepare features and target

#INPUT FEATURES
X = df[["age", "blood_pressure", "cholesterol", "sugar_level"]] 

#TARGET VARIABLE
y = df["disease"]       

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   


# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

#Test the model

y_pred = model.predict(X_test)
print("Predicted:", y_pred)
print("Actual:", y_test.values)
print("Accuracy:", accuracy_score(y_test, y_pred))


# User input section
# ----------------------------
print("Enter patient details")

age = int(input("Enter age(25-80): "))
blood_pressure = int(input("Enter blood pressure(80-200): "))
cholesterol = int(input("Enter cholesterol(150-300): "))
sugar_level = int(input("Enter sugar level(80-200): "))

new_patient = pd.DataFrame([{
    "age": age,
    "blood_pressure": blood_pressure,
    "cholesterol": cholesterol,
    "sugar_level": sugar_level
}])

prediction = model.predict(new_patient)[0]

if prediction == 1:
    print("Disease predicted: YES")
else:
    print("Disease predicted: NO")