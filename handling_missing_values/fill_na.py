import pandas as pd
df=pd.DataFrame({
    "name":[ "manu", "rahul", "anu"],
    "age": [24, None, 20],
    "city": ["bengaluru", "mysore", None]

})
print(df)
#df2=df.fillna("unknown")
#df2=df["age"].fillna(df["age"].mean() )
#df["city"].fillna("DAVANAGERE" )

df2= df.fillna({"age": df["age"].mean(), "city": "DAVANAGERE"}, inplace=True)


print(df2)  