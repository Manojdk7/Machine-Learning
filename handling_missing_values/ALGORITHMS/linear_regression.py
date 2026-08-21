from sklearn.linear_model import LinearRegression
x = [[1], [2], [3], [4], [5]]
y = [40, 50, 60, 70, 80]

model= LinearRegression()
model.fit(x,y)      

hours = float(input("enter the number of hours you studied: "))
predicted_marks = model.predict([[hours]])
print(f" based on the number of hours you studied: {hours}, your marks may be {predicted_marks[0]}")