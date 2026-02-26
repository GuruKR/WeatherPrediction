from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime
import json
import os
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

@app.context_processor
def utility_processor():
    return dict(enumerate=enumerate)


# Database setup
def init_db():
    conn = sqlite3.connect('weather.db')
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create predictions table
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

# Load ML model and utilities
try:
    model = load_model("weather_lstm_model.h5")
    scaler = joblib.load("scaler.pkl")
    label_enc = joblib.load("label_encoder.pkl")
    ml_loaded = True
except:
    ml_loaded = False
    print("Warning: ML model files not found. Running in demo mode.")

# Weather prediction function
def predict_future_weather(temp, humidity, history_buffer):
    if not ml_loaded:
        # Demo mode - random prediction
        conditions = ["Sunny", "Rainy", "Cloudy"]
        return np.random.choice(conditions)
    
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

# Database helper functions
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
        ORDER BY created_at DESC 
        LIMIT ?
    ''', (user_id, limit))
    predictions = cursor.fetchall()
    conn.close()
    return predictions

# Routes
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

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
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials!', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    predictions = get_user_predictions(session['user_id'])
    return render_template('dashboard.html', 
                         username=session['username'],
                         predictions=predictions,
                         ml_loaded=ml_loaded)

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    import requests
    dataa=requests.get("https://api.thingspeak.com/channels/3109146/feeds.json?api_key=KXEBON70KLJQZAOX&results=2")
    vol=dataa.json()['feeds'][-1]['field1']
    cur=dataa.json()['feeds'][-1]['field2']
    temp=dataa.json()['feeds'][-1]['field3']
    hum=dataa.json()['feeds'][-1]['field4']

    
    # Initialize or get buffer from session
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
            
            # Make prediction
            future_weather = predict_future_weather(
                current_temp, 
                current_humidity, 
                session['history_buffer']
            )
            
            if future_weather:
                prediction_result = future_weather
                
                # Determine switching operation
                if future_weather == "Rainy":
                    switching_operation = "Turning ON protective cover system"
                elif future_weather == "Sunny":
                    switching_operation = "Activating solar charging system"
                else:
                    switching_operation = "Keeping system in standby mode"
                
                # Save prediction to database
                save_prediction(
                    session['user_id'],
                    current_temp,
                    current_humidity,
                    future_weather,
                    session['history_buffer']
                )
                
                # Update session buffer
                session.modified = True
                
                flash('Prediction completed successfully!', 'success')
            else:
                flash('Need more historical data for prediction.', 'warning')
                
        except ValueError:
            flash('Please enter valid numerical values.', 'error')
    
    return render_template('prediction.html',
                         prediction_result=prediction_result,
                         switching_operation=switching_operation,
                         buffer_data=session['history_buffer'],
                         ml_loaded=ml_loaded,vol=vol,temp=temp,hum=hum,cur=cur)

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
    flash('Buffer cleared to default values!', 'info')
    return redirect(url_for('prediction'))

@app.route('/api/prediction_stats')
def prediction_stats():
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get prediction counts by weather type
    cursor.execute('''
        SELECT predicted_weather, COUNT(*) as count 
        FROM predictions 
        WHERE user_id = ? 
        GROUP BY predicted_weather
    ''', (session['user_id'],))
    
    stats = cursor.fetchall()
    conn.close()
    
    return jsonify({
        'stats': [dict(row) for row in stats]
    })

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)