from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.DataFrame({

    "size":['small', 'medium', 'large', 'small', 'medium']
}
)


 #create lebel encoder object
le= LabelEncoder()

#apply label encoder on data    
df["size_encoder"]=le.fit_transform(df["size"])
print(df)   