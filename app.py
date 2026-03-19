from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import pickle
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///agritech.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load ML models
model = pickle.load(open('crop_model.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if user already exists
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email address already exists!')
            return redirect(url_for('signup'))
        
        # Create new user
        new_user = User(
            name=name,
            email=email,
            password=generate_password_hash(password, method='pbkdf2:sha256')
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully! Please login.')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('predict'))
        else:
            flash('Invalid email or password!')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            # Get form data in the correct order: N, P, K, temperature, humidity, ph, rainfall
            n = float(request.form.get('n'))
            p = float(request.form.get('p'))
            k = float(request.form.get('k'))
            temperature = float(request.form.get('temperature'))
            humidity = float(request.form.get('humidity'))
            ph = float(request.form.get('ph'))
            rainfall = float(request.form.get('rainfall'))
            
            # Create DataFrame with the exact order expected by the model
            input_data = pd.DataFrame({
                'N': [n],
                'P': [p], 
                'K': [k],
                'temperature': [temperature],
                'humidity': [humidity],
                'ph': [ph],
                'rainfall': [rainfall]
            })
            
            # Make prediction
            prediction_encoded = model.predict(input_data)
            prediction = label_encoder.inverse_transform(prediction_encoded)[0]
            
        except Exception as e:
            flash(f'Error making prediction: {str(e)}')
    
    return render_template('predict.html', prediction=prediction)

@app.route('/account')
@login_required
def account():
    return render_template('account.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
