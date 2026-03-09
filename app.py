from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime
import json
import os
from werkzeug.security import generate_password_hash, check_password_hash
import requests

app = Flask(__name__)
# Load secret from environment for safety (fallback kept for development)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your_secret_key_here')

@app.context_processor
def utility_processor():
    return dict(enumerate=enumerate)


# -------------------------------
# DATABASE SETUP
# -------------------------------
def init_db():
    conn = sqlite3.connect('weather.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            temperature REAL NOT NULL,
            humidity REAL NOT NULL,
            predicted_weather TEXT NOT NULL,
            buffer_data TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()


# -------------------------------
# MODEL LOADING
# -------------------------------
try:
    model = load_model("weather_lstm_model.h5")
    scaler = joblib.load("scaler.pkl")
    label_enc = joblib.load("label_encoder.pkl")
    ml_loaded = True
except Exception as e:
    ml_loaded = False
    print("Warning: ML model failed to load. Running in demo mode.")
    print("Model load error:", e)


# -------------------------------
# WEATHER PREDICTION FUNCTION
# -------------------------------
def predict_future_weather(temp, humidity, history_buffer):
    if not ml_loaded:
        return np.random.choice(["Sunny", "Rainy", "Cloudy"])

    history_buffer.append([temp, humidity])
    
    if len(history_buffer) > 5:
        history_buffer.pop(0)

    if len(history_buffer) < 5:
        return None

    input_seq = scaler.transform(np.array(history_buffer))
    input_seq = np.expand_dims(input_seq, axis=0)
    
    prediction = model.predict(input_seq)
    predicted_label = np.argmax(prediction)
    future_condition = label_enc.inverse_transform([predicted_label])[0]
    
    return future_condition


# -------------------------------
# DATABASE HELPERS
# -------------------------------
def get_db_connection():
    conn = sqlite3.connect('weather.db')
    conn.row_factory = sqlite3.Row
    return conn

def save_prediction(user_id, temperature, humidity, predicted_weather, buffer_data):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO predictions (user_id, temperature, humidity, predicted_weather, buffer_data)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, temperature, humidity, predicted_weather, json.dumps(buffer_data)))
    conn.commit()
    conn.close()

def get_user_predictions(user_id, limit=10):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM predictions
        WHERE user_id = ?
        ORDER BY created_at DESC LIMIT ?
    ''', (user_id, limit))
    predictions = cursor.fetchall()
    conn.close()
    return predictions


# -------------------------------
# ROUTES
# -------------------------------
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')


# -------------------------------
# AUTH
# -------------------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            password_hash = generate_password_hash(password)
            cursor.execute(
                'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                (username, email, password_hash)
            )
            conn.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists!', 'error')
        finally:
            conn.close()
    
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials!', 'error')
    
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


# -------------------------------
# DASHBOARD
# -------------------------------
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    predictions = get_user_predictions(session['user_id'])

    # Compute quick stats server-side so the template has initial values
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT predicted_weather, COUNT(*) as count
        FROM predictions
        WHERE user_id = ?
        GROUP BY predicted_weather
    """, (session['user_id'],))
    rows = cursor.fetchall()
    stats_global = False
    if not rows:
        cursor.execute("""
            SELECT predicted_weather, COUNT(*) as count
            FROM predictions
            GROUP BY predicted_weather
        """)
        rows = cursor.fetchall()
        stats_global = True

    conn.close()

    counts = {'Total': 0, 'Sunny': 0, 'Rainy': 0, 'Cloudy': 0}
    for r in rows:
        if r['predicted_weather'] in counts:
            counts[r['predicted_weather']] = r['count']
            counts['Total'] += r['count']

    # build stats_list for initial chart rendering (always include all categories)
    stats_list = [
        {'predicted_weather': 'Sunny', 'count': counts['Sunny']},
        {'predicted_weather': 'Rainy', 'count': counts['Rainy']},
        {'predicted_weather': 'Cloudy', 'count': counts['Cloudy']},
    ]

    return render_template(
        'dashboard.html',
        username=session['username'],
        predictions=predictions,
        ml_loaded=ml_loaded,
        total_count=counts['Total'],
        sunny_count=counts['Sunny'],
        rainy_count=counts['Rainy'],
        cloudy_count=counts['Cloudy'],
        stats_global=stats_global,
        stats_list=stats_list
    )


# -------------------------------
# WEATHER PREDICTION PAGE
# -------------------------------
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Read ThingSpeak API key from environment (put your key in THINGSPEAK_API_KEY)
    ts_api_key = os.getenv('THINGSPEAK_API_KEY', 'KXEBON70KLJQZAOX')
    ts_url = f"https://api.thingspeak.com/channels/3109146/feeds.json?api_key=KXEBON70KLJQZAOX&results=2"
    dataa = requests.get(ts_url).json()

    vol = dataa['feeds'][-1]['field1']
    cur = dataa['feeds'][-1]['field2']
    temp = dataa['feeds'][-1]['field3']
    hum = dataa['feeds'][-1]['field4']

    if 'history_buffer' not in session:
        session['history_buffer'] = [
            [30.2, 62.5],
            [31.0, 61.0],
            [30.8, 60.0],
            [31.5, 59.2]
        ]
    
    prediction_result = None
    switching_operation = None
    
    if request.method == 'POST':
        try:
            current_temp = float(request.form['temperature'])
            current_humidity = float(request.form['humidity'])

            future_weather = predict_future_weather(
                current_temp,
                current_humidity,
                session['history_buffer']
            )
            
            if future_weather:
                prediction_result = future_weather

                if future_weather == "Rainy":
                    switching_operation = "Turning ON protective cover system"
                elif future_weather == "Sunny":
                    switching_operation = "Activating solar charging system"
                else:
                    switching_operation = "Keeping system in standby mode"

                save_prediction(
                    session['user_id'],
                    current_temp,
                    current_humidity,
                    future_weather,
                    session['history_buffer']
                )

                session.modified = True
            else:
                flash('Need more data for prediction.', 'warning')

        except ValueError:
            flash('Enter valid numbers.', 'error')
    
    return render_template(
        'prediction.html',
        prediction_result=prediction_result,
        switching_operation=switching_operation,
        buffer_data=session['history_buffer'],
        ml_loaded=ml_loaded,
        vol=vol, temp=temp, hum=hum, cur=cur
    )


# -------------------------------
# RESET BUFFER
# -------------------------------
@app.route('/clear_buffer')
def clear_buffer():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    session['history_buffer'] = [
        [30.2, 62.5],
        [31.0, 61.0],
        [30.8, 60.0],
        [31.5, 59.2]
    ]
    session.modified = True
    flash('Buffer reset!', 'info')
    return redirect(url_for('prediction'))


# -------------------------------
# FIXED PREDICTION STATS API
# -------------------------------
@app.route('/api/prediction_stats')
def prediction_stats():
    """Return aggregated counts with optional filters for date range and scope (mine/all/user id)."""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    user_filter = request.args.get('user_id')  # "all", specific id, or None -> current user

    conn = get_db_connection()
    cursor = conn.cursor()

    global_flag = False
    params = []
    conditions = []

    # Date range filter (inclusive)
    if start_date:
        conditions.append("DATE(created_at) >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("DATE(created_at) <= ?")
        params.append(end_date)

    # Determine scope
    if user_filter == "all":
        base_query = """
            SELECT predicted_weather, COUNT(*) as count
            FROM predictions
        """
    elif user_filter and user_filter.isdigit():
        base_query = """
            SELECT predicted_weather, COUNT(*) as count
            FROM predictions
            WHERE user_id = ?
        """
        params.insert(0, int(user_filter))
    elif 'user_id' in session:
        base_query = """
            SELECT predicted_weather, COUNT(*) as count
            FROM predictions
            WHERE user_id = ?
        """
        params.insert(0, session['user_id'])
    else:
        base_query = """
            SELECT predicted_weather, COUNT(*) as count
            FROM predictions
        """

    if conditions:
        if "WHERE" in base_query:
            base_query += " AND " + " AND ".join(conditions)
        else:
            base_query += " WHERE " + " AND ".join(conditions)

    base_query += " GROUP BY predicted_weather"

    cursor.execute(base_query, tuple(params))
    rows = cursor.fetchall()

    # If personal scope is empty and we have a logged-in user, fall back to global
    if (not rows) and ('user_id' in session) and (user_filter not in ("all", None)):
        cursor.execute("""
            SELECT predicted_weather, COUNT(*) as count
            FROM predictions
            GROUP BY predicted_weather
        """)
        rows = cursor.fetchall()
        global_flag = True

    conn.close()

    counts = {'Sunny': 0, 'Rainy': 0, 'Cloudy': 0}
    for row in rows:
        if row["predicted_weather"] in counts:
            counts[row["predicted_weather"]] = row["count"]

    stats = [
        {"predicted_weather": "Sunny", "count": counts["Sunny"]},
        {"predicted_weather": "Rainy", "count": counts["Rainy"]},
        {"predicted_weather": "Cloudy", "count": counts["Cloudy"]},
    ]

    return jsonify({"stats": stats, "global": global_flag})


@app.route('/api/prediction_trend')
def prediction_trend():
    """Return a small recent sequence of predictions for sparkline visuals (max 20)."""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    user_filter = request.args.get('user_id')

    conn = get_db_connection()
    cursor = conn.cursor()

    params = []
    conditions = []

    if start_date:
        conditions.append("DATE(created_at) >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("DATE(created_at) <= ?")
        params.append(end_date)

    if user_filter == "all":
        base_query = "SELECT predicted_weather, created_at FROM predictions"
    elif user_filter and user_filter.isdigit():
        base_query = "SELECT predicted_weather, created_at FROM predictions WHERE user_id = ?"
        params.insert(0, int(user_filter))
    elif 'user_id' in session:
        base_query = "SELECT predicted_weather, created_at FROM predictions WHERE user_id = ?"
        params.insert(0, session['user_id'])
    else:
        base_query = "SELECT predicted_weather, created_at FROM predictions"

    if conditions:
        if "WHERE" in base_query:
            base_query += " AND " + " AND ".join(conditions)
        else:
            base_query += " WHERE " + " AND ".join(conditions)

    base_query += " ORDER BY datetime(created_at) DESC LIMIT 20"

    cursor.execute(base_query, tuple(params))
    rows = cursor.fetchall()
    conn.close()

    # Return in chronological order for charting
    rows = list(rows)[::-1]
    trend = [dict(predicted_weather=r["predicted_weather"], created_at=r["created_at"]) for r in rows]

    return jsonify({"trend": trend})


@app.route('/debug/users')
def debug_users():
    """Debug endpoint (local only) — returns list of users with ids."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT id, username, email, created_at FROM users')
    rows = cur.fetchall()
    conn.close()

    users = []
    for r in rows:
        users.append({
            'id': r['id'],
            'username': r['username'],
            'email': r['email'],
            'created_at': r['created_at']
        })

    return jsonify({'users': users})


# -------------------------------
# RUN APP
# -------------------------------
if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)
