# AI Readmission Risk Copilot

A full-stack AI application that predicts hospital readmission risk using IBM watsonx AutoAI and exposes a real-time API via FastAPI.

## 🔥 Overview

This project demonstrates an end-to-end AI system:

- Train a machine learning model using **IBM watsonx AutoAI**
- Deploy the model as a **real-time API**
- Build a backend service with **FastAPI**
- Generate structured explanations for predictions

## 🧠 Architecture
User Input
↓
FastAPI Backend
↓
IBM watsonx AutoAI (Deployed Model)
↓
Prediction (risk_score, risk_level)
↓
Explanation Layer
↓
Structured JSON Response


## ⚙️ Tech Stack

- **Backend**: FastAPI (Python)
- **AI/ML**: IBM watsonx AutoAI
- **Deployment**: IBM Cloud (watsonx runtime)
- **Data Validation**: Pydantic
- **HTTP Client**: httpx

## 📊 Features

- Predict hospital readmission risk (binary classification)
- Real-time inference using deployed ML model
- Structured output with:
  - risk score
  - risk level
  - key contributing factors
  - recommended actions
- Input validation using Pydantic schemas
- Clean modular backend architecture



### AutoAI Model Training
<img width="1466" height="706" alt="Screenshot 2026-04-11 at 7 19 44 PM" src="https://github.com/user-attachments/assets/c94a252e-1aab-42ec-8407-5dfc47d3ffd6" />

### Model Deployment
<img width="1467" height="744" alt="Screenshot 2026-04-11 at 7 24 18 PM" src="https://github.com/user-attachments/assets/6a8c4098-7230-4fd4-aaf5-2f5dd55c8261" />

### Model Details
<img width="1307" height="640" alt="Screenshot 2026-04-11 at 7 23 59 PM" src="https://github.com/user-attachments/assets/55a4a395-bd8c-4b4f-a199-52982fbb2829" />

## 🚀 How to Run

### 1. Clone the repo

```bash
git clone https://github.com/your-username/ai-readmission-risk-copilot.git
cd ai-readmission-risk-copilot/backend


2. Create virtual environment
python -m venv .venv
source .venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Set environment variables

Create .env file:

WATSONX_API_KEY=your_api_key_here
WATSONX_DEPLOYMENT_URL=your_endpoint_url_here
5. Run the server
uvicorn app.main:app --reload
6. Open API docs
http://127.0.0.1:8000/docs
🧪 Example Request
{
  "age": 68,
  "sex": "female",
  "prior_admissions_12m": 2,
  "length_of_last_stay": 5,
  "comorbidity_count": 3,
  "diabetes": true,
  "hypertension": true,
  "discharge_disposition": "home",
  "follow_up_scheduled": false,
  "medication_adherence_risk": "high",
  "clinical_note": "Patient reports fatigue"
}
📤 Example Response
{
  "risk_score": 0.85,
  "risk_level": "high",
  "summary": "...",
  "key_factors": [...],
  "recommended_actions": [...]
}

💡 Key Learnings
End-to-end AI system design using cloud ML services
Real-world API integration with authentication (IBM IAM)
Handling model input/output schema alignment
Building production-style backend architecture
Importance of data quality in ML pipelines
🔐 Security Note

API keys are managed via environment variables and are not stored in the repository.

👤 Author

Hamzah Ahmadi
