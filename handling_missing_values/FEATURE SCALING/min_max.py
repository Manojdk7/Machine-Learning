from sklearn.preprocessing import MinMaxScaler
import pandas as pd
df=pd.DataFrame({

    "age":[ 21, 22, 20, 19, 18],
    "salary": [10000, 20000, 30000, 40000, 50000]
})          
scaler= MinMaxScaler()
scaled_data= scaler.fit_transform(df)       
print(scaled_data)  