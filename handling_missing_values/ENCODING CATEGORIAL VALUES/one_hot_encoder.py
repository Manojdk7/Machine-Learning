from sklearn.preprocessing import OneHotEncoder
import pandas as pd
df = pd.DataFrame({
    "colors":['red', 'green', 'blue', 'red', 'green']
})  

#creare one hot encoder object
encoder= OneHotEncoder(sparse_output=False)

#apply one hot encoder on data
encoded = encoder.fit_transform(df[["colors"]])

#convert encoded data to dataframe
encoded_df= pd.DataFrame(encoded, columns=encoder.get_feature_names_out(["colors"]))
print(encoded_df)
