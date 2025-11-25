❤️ Heart Disease Prediction Web App

This project is a Flask-based machine learning web app that predicts the risk level of heart disease using patient health data. It also provides personalized lifestyle, medical, and monitoring recommendations — and optionally generates a downloadable PDF health report.

🚀 Features

🧠 Machine Learning Model (Logistic Regression)

📊 Probability-based prediction (Low Risk / High Risk)

📝 Personalized health recommendations

📄 Downloadable PDF report with results

🎨 User-friendly web interface

🔧 Model automatically trains if missing

📂 Tech Stack
Component	Technology
Backend	Flask
ML Model	Logistic Regression (Scikit-Learn)
Data	Synthetic simulated dataset
Frontend	HTML, CSS (inside templates)
Reporting	ReportLab (PDF generation)
▶️ How to Run
# Clone the repo
git clone https://github.com/yourusername/heart-disease-prediction.git

cd heart-disease-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py


Then open in browser:

http://localhost:8080

📁 Project Structure
|-- app.py
|-- templates/
|   |-- index.html
|   |-- result.html
|   |-- error.html
|-- model.pkl (auto-generated)
|-- scaler.pkl (auto-generated)

⚠️ Disclaimer

This application is for educational and research purposes only.
It is not a medical diagnostic tool and should not replace professional healthcare advice.

⭐ Contribution

Pull requests and feature improvements are welcome!
