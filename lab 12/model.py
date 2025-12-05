# model_creation.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# 1. Load data
df = pd.read_csv("used_cars.csv")

# 2. Clean price and mileage
df['price'] = df['price'].replace('[\$,]', '', regex=True).astype(float)
df['milage'] = df['milage'].str.replace(' mi.', '', regex=False).str.replace(',', '').astype(int)

# 3. Handle missing values
df['fuel_type'].fillna('Unknown', inplace=True)
df['accident'].fillna('Unknown', inplace=True)
df.dropna(subset=['clean_title'], inplace=True)
df.reset_index(drop=True, inplace=True)

# 4. Encode categorical columns
categorical_cols = ['brand', 'model', 'fuel_type', 'engine', 'transmission',
                    'ext_col', 'int_col', 'accident', 'clean_title']

label_encoders = {}
df_encoded = df.copy()

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = df_encoded[col].astype(str)
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# 5. Prepare features (X) and target (y)
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Scale features
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# 8. Train SVR model
model = SVR(kernel='rbf', C=100, gamma=0.01, epsilon=1000)
model.fit(X_train_scaled, y_train_scaled)

# 9. Predict and evaluate
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model trained successfully!")
print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"R² Score: {r2:.4f}")

# 10. Save model and preprocessors
pickle.dump(model, open('svr_model.pkl', 'wb'))
pickle.dump(scaler_X, open('scaler_X.pkl', 'wb'))
pickle.dump(scaler_y, open('scaler_y.pkl', 'wb'))
pickle.dump(label_encoders, open('label_encoders.pkl', 'wb'))

print("Model and preprocessors saved!")