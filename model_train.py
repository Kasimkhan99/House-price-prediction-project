import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib.pyplot as plt




data = pd.read_csv('house_data.csv')

#print(data.head())
#print(data.isna().sum())

#print(data.describe())

x= data[['area','bedrooms','bathrooms','stories','parking','year_built','city']]
y=data['price']

x['house_age']=2025-x['year_built']
x['area_per_bedroom'] = x['area'] / x['bedrooms'].replace(0, np.nan)
x['area_per_bedroom'] = x['area_per_bedroom'].fillna(x['area'])


x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.2,random_state=42)

#print(x.head())
numeric_feature=['area','bathrooms','stories','parking','year_built','house_age','area_per_bedroom']
categorical_feature=['city']

numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())

])
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor=ColumnTransformer(
    transformers=[
        ('num',numeric_pipeline,numeric_feature),
        ('cat',categorical_pipeline,categorical_feature)
    ]
)

model=Pipeline(steps=[
    ('preprocess',preprocessor),
    ('regressor',RandomForestRegressor(n_estimators=300,random_state=42))
]

)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

#accuracy=accuracy_score(y_test,y_pred) accuracy score linear algo ma nhi niklta ye sirf classification ma niklta ha

mean=mean_absolute_error(y_test,y_pred)
rmse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

print("mean" ,mean)
print("rmse" , rmse)
print("r2",r2)


plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred ,alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', lw=2) 

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")


plt.show()


import joblib

joblib.dump(model,'house_price_model.pkl')