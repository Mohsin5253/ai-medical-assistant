import os
import joblib
import numpy as np
from datetime import datetime
import io
from dotenv import load_dotenv
load_dotenv() 

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from groq import Groq

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_RIGHT

app = Flask(__name__)

app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "fallback-dev-key-123")
db_url = os.environ.get('DATABASE_URL', 'sqlite:////tmp/disease_predictor.db')
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login' 
login_manager.login_message_category = 'info'

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'ml_models/rf_model.pkl')
SYMPTOMS_PATH = os.path.join(BASE_DIR, 'ml_models/symptoms_list.pkl')

model = None
symptoms_list = None

if os.path.exists(MODEL_PATH) and os.path.exists(SYMPTOMS_PATH):
    model = joblib.load(MODEL_PATH)
    symptoms_list = joblib.load(SYMPTOMS_PATH)
    print(f"✅ ML Model loaded safely. Tracking {len(symptoms_list)} symptoms.")
else:
    print("⚠️ Warning: ML Models not found. Run train_model.py first.")

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    history = db.relationship('PredictionHistory', backref='patient', lazy=True)

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    disease = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    symptoms = db.Column(db.Text, nullable=False)
    llm_advice = db.Column(db.Text, nullable=True)
    date_predicted = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    
    return db.session.get(User, int(user_id))

with app.app_context():
    db.create_all()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        email_input = request.form.get('email')
        username_input = request.form.get('username')
        
        # 1. Check if email already exists
        existing_email = User.query.filter_by(email=email_input).first()
        if existing_email:
            flash('That email is already registered. Please log in.', 'danger')
            return redirect(url_for('register'))
            
        # 2. Check if username already exists
        existing_username = User.query.filter_by(username=username_input).first()
        if existing_username:
            flash('That username is already taken. Please choose another.', 'danger')
            return redirect(url_for('register'))

        # 3. If both are unique, create the account safely
        hashed_password = bcrypt.generate_password_hash(request.form.get('password')).decode('utf-8')
        user = User(username=username_input, email=email_input, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        
        flash('Account created successfully! You can now log in.', 'success')
        return redirect(url_for('login'))
        
    return render_template('register.html')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        user = User.query.filter_by(email=request.form.get('email')).first()
        if user and bcrypt.check_password_hash(user.password, request.form.get('password')):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required 
def dashboard():
    formatted_symptoms = []
    if symptoms_list:
        for sym in sorted(symptoms_list):
            formatted_symptoms.append({'original': sym, 'display': sym.replace('_', ' ').title()})
    return render_template('dashboard.html', symptoms=formatted_symptoms)

@app.route('/history')
@login_required
def history():
    records = PredictionHistory.query.filter_by(user_id=current_user.id).order_by(PredictionHistory.date_predicted.desc()).all()
    return render_template('history.html', records=records)

@app.route('/api/predict', methods=['POST'])
@login_required
def predict_api():
    if not model or not symptoms_list:
        return jsonify({'error': 'Model not loaded.'}), 500

    try:
        data = request.get_json()
        selected_symptoms = data.get('symptoms', [])
        
        if not selected_symptoms:
            return jsonify({'error': 'Please select at least one symptom.'}), 400
            
        input_data = np.zeros(len(symptoms_list))
        display_symptoms = []
        for symptom in selected_symptoms:
            if symptom in symptoms_list:
                index = symptoms_list.index(symptom)
                input_data[index] = 1
                display_symptoms.append(symptom.replace('_', ' ').title())
                
        features_array = input_data.reshape(1, -1)
        prediction = model.predict(features_array)[0]
        confidence_score = round(max(model.predict_proba(features_array)[0]) * 100, 2)
        
        contributing_symptoms = []
        try:
            importances = model.feature_importances_
            for i, symptom in enumerate(symptoms_list):
                if input_data[i] == 1 and importances[i] > 0:
                    impact_score = round(float(importances[i]) * 1000, 2)
                    contributing_symptoms.append({
                        'symptom': symptom.replace('_', ' ').title(), 
                        'impact': impact_score
                    })
            contributing_symptoms = sorted(contributing_symptoms, key=lambda x: x['impact'], reverse=True)
        except Exception as e:
            print("Explainability Error:", e)

        llm_response = "Explanation could not be generated. Please configure your Groq API key."
        if groq_client:
            prompt = f"""
            A patient has reported the following symptoms: {', '.join(display_symptoms)}.
            An AI model has predicted they might have: {prediction}.
            
            Please provide a response structured exactly like this:
            
            Disease Overview:
            Write a clear, compassionate, and comprehensive 3 to 4 sentence explanation of what {prediction} is and how it relates to their specific symptoms. 
            
            Lifestyle & Care Recommendations:
            Provide 3 highly specific, actionable, and helpful lifestyle or first-aid recommendations tailored specifically for someone dealing with {prediction}. Format these as a bulleted list.
            
            
            [LEAVE A BLANK LINE HERE]
            
            MEDICAL DISCLAIMER: Write a strong, fully capitalized disclaimer stating that this is an AI-generated prediction, not a medical diagnosis, and they MUST consult a qualified healthcare professional immediately.
            """
            try:
                chat_completion = groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are an expert, empathetic medical AI assistant. You structure your responses cleanly and provide highly relevant, accurate medical context."},
                        {"role": "user", "content": prompt}
                    ],
                    model="llama-3.1-8b-instant",
                    temperature=0.7,
                    max_tokens=1000, # Increased from 300 to allow for the longer, nicer explanations
                )
                llm_response = chat_completion.choices[0].message.content
            except Exception as e:
                print(f"Groq API Error: {e}")
                llm_response = "Explanation could not be generated at this time. Please consult a doctor immediately."

        new_record = PredictionHistory(
            user_id=current_user.id,
            disease=prediction,
            confidence=confidence_score,
            symptoms=', '.join(display_symptoms),
            llm_advice=llm_response
        )
        db.session.add(new_record)
        db.session.commit()

        return jsonify({
            'disease': prediction,
            'confidence': confidence_score,
            'contributing_factors': contributing_symptoms,
            'llm_explanation': llm_response,
            'history_id': new_record.id,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/download_pdf/<int:history_id>')
@login_required
def download_pdf(history_id):
    record = PredictionHistory.query.get_or_404(history_id)
    if record.user_id != current_user.id:
        return redirect(url_for('dashboard'))

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    story = []

    story.append(Paragraph("<b>AI Medical Prediction Report</b>", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"<b>Patient:</b> {current_user.username}", styles['Normal']))
    
    # Date Generated formatted to "29 June 2026, 04:30 PM"
    real_time_now = datetime.now().strftime('%d %B %Y, %I:%M %p')
    story.append(Paragraph(f"<b>Date Generated:</b> {real_time_now}", styles['Normal']))
    
    story.append(Spacer(1, 12))
    
    story.append(Paragraph(f"<b>Reported Symptoms:</b> {record.symptoms}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph(f"<b>Predicted Condition:</b> {record.disease} (Confidence: {record.confidence}%)", styles['Heading2']))
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("<b>AI Doctor's Notes:</b>", styles['Heading3']))
    
    # Fix the spacing: Remove markdown asterisks AND convert newlines to <br/> tags for the PDF
    clean_llm = record.llm_advice.replace('*', '').replace('\n', '<br/>') 
    story.append(Paragraph(clean_llm, styles['Normal']))

    doc.build(story)
    buffer.seek(0)
    
    return send_file(
        buffer, 
        as_attachment=True, 
        download_name=f"Medical_Report_{record.disease}.pdf", 
        mimetype='application/pdf'
    )

if __name__ == '__main__':
    with app.app_context():
        db.create_all() 
        print("✅ Database verified.")
        
    print("🚀 Server is starting! Open your browser and go to: http://127.0.0.1:8000")
    
    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=False)