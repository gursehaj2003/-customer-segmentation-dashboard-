import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

X_train = pd.read_csv('X_train.csv').values
y_train = pd.read_csv('y_train.csv').values.ravel()
X_test = pd.read_csv('X_test.csv').values
y_test = pd.read_csv('y_test.csv').values.ravel()

rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

joblib.dump(rf, 'churn_rf_model.pkl')
pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).to_csv('churn_predictions.csv', index=False)

print("✅ Churn model trained!")
print(f"Accuracy: {sum(y_test==y_pred)/len(y_test):.2%}")
