from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
import io
app = Flask(__name__)

# Create comprehensive heart disease dataset with additional features
def create_sample_data():
    np.random.seed(42)
    n_samples = 1000

    # Basic features
    data = {
        'age': np.random.randint(29, 77, n_samples),
        'sex': np.random.randint(0, 2, n_samples),  # 0=female, 1=male
        'cp': np.random.randint(0, 4, n_samples),   # chest pain type
        'trestbps': np.random.randint(94, 200, n_samples),  # resting blood pressure
        'chol': np.random.randint(126, 564, n_samples),     # serum cholesterol
        'fbs': np.random.randint(0, 2, n_samples),          # fasting blood sugar
        'restecg': np.random.randint(0, 3, n_samples),      # resting electrocardiographic results
        'thalach': np.random.randint(71, 202, n_samples),   # maximum heart rate achieved
        'exang': np.random.randint(0, 2, n_samples),        # exercise induced angina
        'oldpeak': np.random.uniform(0, 6.2, n_samples),    # ST depression induced by exercise
        'slope': np.random.randint(0, 3, n_samples),        # slope of peak exercise ST segment
        'ca': np.random.randint(0, 4, n_samples),           # number of major vessels colored by fluoroscopy
        'thal': np.random.randint(0, 3, n_samples),         # thalassemia

        # Additional advanced features
        'smoking': np.random.randint(0, 2, n_samples),      # smoking status
        'alcohol': np.random.randint(0, 2, n_samples),      # alcohol consumption
        'bmi': np.random.uniform(18.5, 40, n_samples),      # body mass index
        'family_history': np.random.randint(0, 2, n_samples), # family history of heart disease
        'diabetes': np.random.randint(0, 2, n_samples),     # diabetes status
        'hypertension': np.random.randint(0, 2, n_samples), # hypertension status
        'stress_level': np.random.randint(1, 6, n_samples), # stress level (1-5)
        'exercise_freq': np.random.randint(0, 8, n_samples), # exercise frequency per week
        'sleep_hours': np.random.uniform(4, 12, n_samples), # average sleep hours
        'diet_score': np.random.randint(1, 11, n_samples),  # diet quality score (1-10)
    }

    df = pd.DataFrame(data)
    # Target: heart disease (0=no, 1=yes)
    df['target'] = np.random.randint(0, 2, n_samples)

    return df

# Train and save model
def train_model():
    df = create_sample_data()

    # Use only the 13 basic features for the model (same as original)
    basic_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                     'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

    X = df[basic_features]
    y = df['target']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Save model and scaler
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return model, scaler

# Load model and scaler
def load_model():
    if not os.path.exists('model.pkl') or not os.path.exists('scaler.pkl'):
        return train_model()

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    return model, scaler

# Generate recommendations based on risk factors
def generate_recommendations(form_data, risk_level, probability):
    recommendations = {
        'immediate': [],
        'lifestyle': [],
        'medical': [],
        'monitoring': []
    }

    # Parse form data
    age = int(form_data.get('age', 0))
    sex = form_data.get('sex', '0')
    smoking = form_data.get('smoking', '0')
    alcohol = form_data.get('alcohol', '0')
    bmi = float(form_data.get('bmi', 25))
    exercise_freq = int(form_data.get('exercise_freq', 0))
    diet_score = int(form_data.get('diet_score', 5))
    stress_level = int(form_data.get('stress_level', 3))
    sleep_hours = float(form_data.get('sleep_hours', 7))
    family_history = form_data.get('family_history', '0')
    diabetes = form_data.get('diabetes', '0')
    hypertension = form_data.get('hypertension', '0')

    # Immediate actions for high risk
    if risk_level == "High Risk":
        recommendations['immediate'].append("Schedule an appointment with a cardiologist within the next 7 days")
        recommendations['immediate'].append("Consider emergency room visit if experiencing chest pain, shortness of breath, or dizziness")
        recommendations['immediate'].append("Start daily blood pressure and heart rate monitoring")

    # Lifestyle recommendations
    if smoking == '1':
        recommendations['lifestyle'].append("Quit smoking immediately - this is the single most important step you can take")
        recommendations['lifestyle'].append("Join a smoking cessation program or consider nicotine replacement therapy")

    if alcohol == '1':
        recommendations['lifestyle'].append("Limit alcohol consumption to moderate levels (1 drink/day for women, 2 for men)")
        recommendations['lifestyle'].append("Consider complete abstinence if you have other risk factors")

    if bmi > 25:
        recommendations['lifestyle'].append("Aim for gradual weight loss of 1-2 pounds per week")
        recommendations['lifestyle'].append("Consult a registered dietitian for personalized meal planning")
        recommendations['lifestyle'].append("Increase physical activity to 150 minutes of moderate exercise per week")

    if exercise_freq < 3:
        recommendations['lifestyle'].append("Start with 30 minutes of moderate exercise most days of the week")
        recommendations['lifestyle'].append("Include both aerobic exercise (walking, swimming) and strength training")
        recommendations['lifestyle'].append("Consult your doctor before starting a new exercise program")

    if diet_score < 7:
        recommendations['lifestyle'].append("Adopt a heart-healthy diet rich in fruits, vegetables, whole grains, and lean proteins")
        recommendations['lifestyle'].append("Reduce saturated fats, trans fats, and cholesterol intake")
        recommendations['lifestyle'].append("Limit sodium to 2,300 mg per day (1,500 mg if you have hypertension)")

    if stress_level > 3:
        recommendations['lifestyle'].append("Practice stress-reduction techniques like meditation, yoga, or deep breathing")
        recommendations['lifestyle'].append("Ensure adequate sleep (7-9 hours per night)")
        recommendations['lifestyle'].append("Consider counseling or support groups for stress management")

    if sleep_hours < 7:
        recommendations['lifestyle'].append("Establish a regular sleep schedule and bedtime routine")
        recommendations['lifestyle'].append("Create a sleep-friendly environment (cool, dark, quiet)")
        recommendations['lifestyle'].append("Avoid screens and caffeine before bedtime")

    # Medical recommendations
    if family_history == '1':
        recommendations['medical'].append("Regular cardiac screenings every 1-2 years")
        recommendations['medical'].append("Genetic counseling if multiple family members affected")

    if diabetes == '1':
        recommendations['medical'].append("Maintain tight blood sugar control (HbA1c < 7%)")
        recommendations['medical'].append("Regular eye and kidney function monitoring")

    if hypertension == '1':
        recommendations['medical'].append("Take prescribed blood pressure medications consistently")
        recommendations['medical'].append("Monitor blood pressure at home regularly")

    # Monitoring recommendations
    recommendations['monitoring'].append("Regular check-ups with your primary care physician")
    recommendations['monitoring'].append("Annual lipid profile and blood glucose testing")
    recommendations['monitoring'].append("Periodic ECG and stress testing as recommended by your doctor")
    recommendations['monitoring'].append("Keep a personal health journal tracking symptoms and medications")

    return recommendations

# Generate PDF report
def generate_pdf_report(form_data, prediction, probability, recommendations):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = styles['Heading1']
    title_style.alignment = 1  # Center alignment
    story.append(Paragraph("Heart Disease Risk Assessment Report", title_style))
    story.append(Spacer(1, 12))

    # Report date
    story.append(Paragraph(f"Report Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Patient Information Section
    story.append(Paragraph("Patient Information", styles['Heading2']))
    story.append(Spacer(1, 6))

    patient_data = [
        ["Age", form_data.get('age', 'N/A')],
        ["Sex", "Female" if form_data.get('sex') == '0' else "Male"],
        ["BMI", f"{form_data.get('bmi', 'N/A')} kg/m²"],
        ["Smoking Status", "Yes" if form_data.get('smoking') == '1' else "No"],
        ["Alcohol Consumption", "Yes" if form_data.get('alcohol') == '1' else "No"],
        ["Exercise Frequency", f"{form_data.get('exercise_freq', 'N/A')} days/week"],
        ["Diet Score", f"{form_data.get('diet_score', 'N/A')}/10"],
        ["Stress Level", f"{form_data.get('stress_level', 'N/A')}/5"],
        ["Sleep Hours", f"{form_data.get('sleep_hours', 'N/A')} hours/night"],
    ]

    patient_table = Table(patient_data, colWidths=[200, 200])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 12))

    # Risk Assessment Section
    story.append(Paragraph("Risk Assessment Results", styles['Heading2']))
    story.append(Spacer(1, 6))

    risk_color = colors.red if prediction == "High Risk" else colors.green
    risk_data = [
        ["Risk Level", prediction],
        ["Probability", f"{probability:.1%}"],
        ["Assessment Date", datetime.now().strftime('%B %d, %Y')]
    ]

    risk_table = Table(risk_data, colWidths=[200, 200])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('TEXTCOLOR', (1, 1), (1, 1), risk_color),
        ('FONTNAME', (1, 1), (1, 1), 'Helvetica-Bold')
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 12))

    # Recommendations Section
    story.append(Paragraph("Personalized Recommendations", styles['Heading2']))
    story.append(Spacer(1, 6))

    if recommendations['immediate']:
        story.append(Paragraph("Immediate Actions Required:", styles['Heading3']))
        for rec in recommendations['immediate']:
            story.append(Paragraph(f"• {rec}", styles['Normal']))
        story.append(Spacer(1, 6))

    if recommendations['medical']:
        story.append(Paragraph("Medical Recommendations:", styles['Heading3']))
        for rec in recommendations['medical']:
            story.append(Paragraph(f"• {rec}", styles['Normal']))
        story.append(Spacer(1, 6))

    if recommendations['lifestyle']:
        story.append(Paragraph("Lifestyle Modifications:", styles['Heading3']))
        for rec in recommendations['lifestyle']:
            story.append(Paragraph(f"• {rec}", styles['Normal']))
        story.append(Spacer(1, 6))

    if recommendations['monitoring']:
        story.append(Paragraph("Monitoring & Follow-up:", styles['Heading3']))
        for rec in recommendations['monitoring']:
            story.append(Paragraph(f"• {rec}", styles['Normal']))
        story.append(Spacer(1, 6))

    # Disclaimer
    story.append(Spacer(1, 12))
    story.append(Paragraph("Important Disclaimer", styles['Heading3']))
    disclaimer_text = """
    This report is generated by an AI-powered system for informational purposes only.
    It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
    Always seek the advice of your physician or other qualified health provider with any questions
    you may have regarding a medical condition. Never disregard professional medical advice or delay
    in seeking it because of something you have read in this report.
    """
    story.append(Paragraph(disclaimer_text, styles['Normal']))

    doc.build(story)
    buffer.seek(0)
    return buffer

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get all form data
        form_data = request.form.to_dict()

        # Validate required fields
        required_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                          'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

        for field in required_fields:
            if field not in form_data or not form_data[field].strip():
                return render_template('error.html',
                                     error=f"Missing required field: {field}")

        # Safely get values with defaults
        def safe_int(value, default=0):
            try:
                return int(float(value)) if value else default
            except (ValueError, TypeError):
                return default

        def safe_float(value, default=0.0):
            try:
                return float(value) if value else default
            except (ValueError, TypeError):
                return default

        # Basic features for model (13 original features)
        features = [
            safe_float(form_data['age']),
            safe_int(form_data['sex']),
            safe_int(form_data['cp']),
            safe_float(form_data['trestbps']),
            safe_float(form_data['chol']),
            safe_int(form_data['fbs']),
            safe_int(form_data['restecg']),
            safe_float(form_data['thalach']),
            safe_int(form_data['exang']),
            safe_float(form_data['oldpeak']),
            safe_int(form_data['slope']),
            safe_int(form_data['ca']),
            safe_int(form_data['thal'])
        ]

        # Load model and scaler
        model, scaler = load_model()

        # Scale features
        features_scaled = scaler.transform([features])

        # Make prediction
        prediction_num = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        prediction = "High Risk" if prediction_num == 1 else "Low Risk"

        # Generate recommendations
        recommendations = generate_recommendations(form_data, prediction, probability)

        return render_template('result.html',
                             prediction=prediction,
                             probability=f"{probability:.2%}",
                             form_data=form_data,
                             recommendations=recommendations)

    except Exception as e:
        return render_template('error.html', error=f"Prediction failed: {str(e)}")



@app.route('/download_report', methods=['POST'])
def download_report():
    try:
        form_data = request.form.to_dict()

        # Extract prediction data from form
        prediction = form_data.get('prediction', 'Unknown')
        probability_str = form_data.get('probability', '0%')
        probability = float(probability_str.strip('%')) / 100

        # Generate recommendations
        recommendations = generate_recommendations(form_data, prediction, probability)

        # Generate PDF
        pdf_buffer = generate_pdf_report(form_data, prediction, probability, recommendations)

        # Return PDF as downloadable file
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=f'heart_disease_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
            mimetype='application/pdf'
        )

    except Exception as e:
        return render_template('error.html', error=f"PDF generation failed: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
