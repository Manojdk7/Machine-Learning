import pandas as pd
df=pd.DataFrame({
    "name":[ "manu", "rahul", "anu"],
    "age": [21, None, 20],
    "city": ["bengaluru", "mysore", None]

})
print(df)




print(df.dropna)

print(df.dropna(axis=1))

print(df.dropna(axis=0, how='any'))