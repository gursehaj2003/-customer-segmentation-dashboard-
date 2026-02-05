import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('online_retail_customer_data_extended.csv')
df = df.dropna()

le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])
le_region = LabelEncoder()
df['Region'] = le_region.fit_transform(df['Region'])
le_membership = LabelEncoder()
df['Membership_Status'] = le_membership.fit_transform(df['Membership_Status'])

seg_features = ['Age', 'Annual_Income_USD', 'Spending_Score', 'Total_Purchases', 'Avg_Purchase_Value', 'Satisfaction_Score']
churn_features = seg_features + ['Gender', 'Region', 'Website_Visits_Last_Month']

scaler_seg = StandardScaler()
X_seg_scaled = scaler_seg.fit_transform(df[seg_features])

scaler_churn = StandardScaler()
X_churn_scaled = scaler_churn.fit_transform(df[churn_features])
y_churn = df['Churn'].values

X_train, X_test, y_train, y_test = train_test_split(X_churn_scaled, y_churn, test_size=0.2, random_state=42)

pd.DataFrame(X_seg_scaled, columns=seg_features).to_csv('segmentation_data.csv', index=False)
pd.DataFrame(X_train).to_csv('X_train.csv', index=False)
pd.DataFrame(y_train).to_csv('y_train.csv', index=False)
pd.DataFrame(X_test).to_csv('X_test.csv', index=False)
pd.DataFrame(y_test).to_csv('y_test.csv', index=False)
df.to_csv('full_data.csv', index=False)

print("✅ Data prepared!")
