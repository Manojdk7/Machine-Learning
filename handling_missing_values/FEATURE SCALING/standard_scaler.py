from sklearn.preprocessing import StandardScaler
import pandas as pd
df=pd.DataFrame({
    "age":[ 21, 22, 20, 19, 18],
    "salary": [10000, 20000, 30000, 40000, 50000]
})          

scaler=  StandardScaler()
scaled_data= scaler.fit_transform(df)       
print(scaled_data)
