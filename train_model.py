import pandas as pd

# Load the cleaned combined dataset
df = pd.read_csv('data/heart_combined_cleaned.csv')

# Strip any accidental spaces from column names
df.columns = df.columns.str.strip()

# Check column names
print(df.columns)

categorical_cols = ['cp', 'restecg', 'slope', 'thal', 'sex', 'fbs', 'exang']
for col in categorical_cols:
    if col not in df.columns:
        print(f"Warning: {col} not in dataset")

# Apply get_dummies only to the columns that exist
df = pd.get_dummies(df, columns=[col for col in categorical_cols if col in df.columns], drop_first=True)
# Drop unnecessary columns
X = df.drop(['id', 'dataset', 'num'], axis=1)

# Target: convert 'num' to binary (1 = risk, 0 = no risk)
y = df['num'].apply(lambda x: 1 if x > 0 else 0)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Numeric columns to scale
numeric_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
from sklearn.ensemble import RandomForestClassifier
import joblib

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)
joblib.dump(model, 'model/heart_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(X.columns.tolist(), 'model/columns.pkl')

print("Model, scaler, and columns saved successfully!")
