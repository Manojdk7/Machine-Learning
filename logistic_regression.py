

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

base_dir = Path(__file__).resolve().parent
data_path = base_dir / "Housing.csv"
df = pd.read_csv(data_path, sep="\t")

median_price = df["price"].median()
df["price_label"] = (df["price"] > median_price).astype(int)

X = df.drop(columns=["price", "price_label"])
y = df["price_label"]

X["mainroad"] = X["mainroad"].map({"yes": 1, "no": 0})
X["guestroom"] = X["guestroom"].map({"yes": 1, "no": 0})
X["basement"] = X["basement"].map({"yes": 1, "no": 0})
X["hotwaterheating"] = X["hotwaterheating"].map({"yes": 1, "no": 0})
X["airconditioning"] = X["airconditioning"].map({"yes": 1, "no": 0})
X["prefarea"] = X["prefarea"].map({"yes": 1, "no": 0})
X["furnishingstatus"] = X["furnishingstatus"].map({
    "furnished": 2,
    "semi-furnished": 1,
    "unfurnished": 0
})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Enter house features:")
area = int(input("Enter area: "))
bedrooms = int(input("Enter bedrooms: "))
bathrooms = int(input("Enter bathrooms: "))
stories = int(input("Enter stories: "))
mainroad = input("Main road? (yes/no): ").strip().lower()
guestroom = input("Guestroom? (yes/no): ").strip().lower()
basement = input("Basement? (yes/no): ").strip().lower()
hotwaterheating = input("Hot water heating? (yes/no): ").strip().lower()
airconditioning = input("Air conditioning? (yes/no): ").strip().lower()
parking = int(input("Parking spaces: "))
prefarea = input("Prefarea? (yes/no): ").strip().lower()
furnishingstatus = input("Furnishing status (furnished/semi-furnished/unfurnished): ").strip().lower()

new_house = pd.DataFrame([{
    "area": area,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "mainroad": mainroad,
    "guestroom": guestroom,
    "basement": basement,
    "hotwaterheating": hotwaterheating,
    "airconditioning": airconditioning,
    "parking": parking,
    "prefarea": prefarea,
    "furnishingstatus": furnishingstatus
}])

new_house["mainroad"] = new_house["mainroad"].map({"yes": 1, "no": 0})
new_house["guestroom"] = new_house["guestroom"].map({"yes": 1, "no": 0})
new_house["basement"] = new_house["basement"].map({"yes": 1, "no": 0})
new_house["hotwaterheating"] = new_house["hotwaterheating"].map({"yes": 1, "no": 0})
new_house["airconditioning"] = new_house["airconditioning"].map({"yes": 1, "no": 0})
new_house["prefarea"] = new_house["prefarea"].map({"yes": 1, "no": 0})
new_house["furnishingstatus"] = new_house["furnishingstatus"].map({
    "furnished": 2,
    "semi-furnished": 1,
    "unfurnished": 0
})

prediction = model.predict(new_house)[0]

if prediction == 1:
    print("This house is predicted as EXPENSIVE.")
else:
    print("This house is predicted as NOT EXPENSIVE.")