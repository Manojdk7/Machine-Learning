import pandas as pd
df=pd.DataFrame({
    "name":[ "manu", "rahul", "anu"],
    "age": [21, None, 20]

})
print(df)
print(df.isnull().sum())
print(df.isnull())
