from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# 1. Load the Breast Cancer dataset
data = load_breast_cancer()

X = data.data
feature_names = data.feature_names

# Original labels:
# 0 = malignant
# 1 = benign
#
# We convert the target so:
# 1 = malignant
# 0 = benign
y = (data.target == 0).astype(int)

print("Dataset shape:", X.shape)
print("Number of features:", X.shape[1])
print("Feature names:", feature_names)
print("Classes: 1 = Malignant, 0 = Benign")

# 2. Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# 3. Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=None,
    random_state=42,
    class_weight="balanced",
)

# 4. Train the model
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate the model
print("\n=== Model Performance ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred, zero_division=0):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Benign", "Malignant"],
        zero_division=0,
    )
)

# 7. Display the most important features
feature_importance = sorted(
    zip(feature_names, model.feature_importances_),
    key=lambda item: item[1],
    reverse=True,
)

print("\nTop 10 Important Features:")
for feature, importance in feature_importance[:10]:
    print(f"{feature}: {importance:.4f}")

# 8. Predict one test sample
sample = X_test[0].reshape(1, -1)
prediction = model.predict(sample)[0]
probability = model.predict_proba(sample)[0][1]

print("\n=== Example Prediction ===")
if prediction == 1:
    print("Prediction: Malignant")
else:
    print("Prediction: Benign")

print(f"Probability of malignant class: {probability:.4f}")