import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for, session
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash

# -----------------------------
# Load model, scaler, and columns
# -----------------------------
model = joblib.load('model/heart_model.pkl')
scaler = joblib.load('model/scaler.pkl')
columns = joblib.load('model/columns.pkl')  # saved training columns

# Load cleaned dataset for dashboard
df = pd.read_csv('data/heart_combined_cleaned.csv')

# -----------------------------
# Flask setup
# -----------------------------
app = Flask(__name__)
app.secret_key = "your_secret_key"

# MySQL Connection
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Nisha@04",
    database="heart_app"
)
cursor = conn.cursor(dictionary=True)

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')

# Signup
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        try:
            cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                           (username, email, password))
            conn.commit()
            return redirect(url_for('login'))
        except:
            return "Username or Email already exists!"
    return render_template('signup.html')

# Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
        user = cursor.fetchone()
        if user and check_password_hash(user['password'], password):
            session['user'] = user['username']
            return redirect(url_for('dashboard'))  # Redirect to dashboard after login
        else:
            return "Invalid Credentials!"
    return render_template('login.html')

# Dashboard
@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))

    total = len(df)
    male = len(df[df['sex'] == 'Male'])
    female = len(df[df['sex'] == 'Female'])
    # Count patients with heart attack risk (num > 0)
    risk = len(df[df['num'] > 0])

    return render_template('dashboard.html', total=total, male=male, female=female, risk=risk)

# Home / Prediction
@app.route('/home', methods=['GET', 'POST'])
def home():
    if 'user' not in session:
        return redirect(url_for('login'))

    prediction = None

    if request.method == 'POST':
        # -----------------------
        # Collect numeric inputs
        # -----------------------
        age = float(request.form['age'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        thalch = float(request.form['thalch'])
        oldpeak = float(request.form['oldpeak'])
        ca = float(request.form['ca'])

        # -----------------------
        # Collect binary inputs
        # -----------------------
        sex = 1 if request.form['sex'] == 'Male' else 0
        fbs = 1 if request.form['fbs'] == 'True' else 0
        exang = 1 if request.form['exang'] == 'True' else 0

        # -----------------------
        # Collect categorical inputs
        # -----------------------
        cp = request.form['cp']
        restecg = request.form['restecg']
        slope = request.form['slope']
        thal = request.form['thal']

        # -----------------------
        # Create DataFrame for prediction
        # -----------------------
        input_dict = {
            'age': [age],
            'trestbps': [trestbps],
            'chol': [chol],
            'thalch': [thalch],
            'oldpeak': [oldpeak],
            'ca': [ca],
            'sex': [sex],
            'fbs': [fbs],
            'exang': [exang]
        }

        # Initialize all categorical columns with 0
        for col in columns:
            if col not in input_dict:
                input_dict[col] = [0]

        # One-hot encode the categorical inputs
        input_dict[f'cp_{cp}'] = [1]
        input_dict[f'restecg_{restecg}'] = [1]
        input_dict[f'slope_{slope}'] = [1]
        input_dict[f'thal_{thal}'] = [1]

        # Convert to DataFrame
        X = pd.DataFrame(input_dict)

        # Ensure the order and all columns exist
        X = X.reindex(columns=columns, fill_value=0)

        # Scale numeric columns
        numeric_cols = ['age','trestbps','chol','thalch','oldpeak','ca']
        X[numeric_cols] = scaler.transform(X[numeric_cols])

        # Predict
        pred = model.predict(X)[0]
        prediction = "Heart Attack Risk" if pred > 0 else "No Heart Attack Risk"

    return render_template('home.html', prediction=prediction)

# Logout
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

# -----------------------------
# Run Flask app
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)
