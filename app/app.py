import os
import sys
import streamlit as st

# Inject src folder into Python path so imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'src'))

# Import the inference function
try:
    from src.predict import predict_loan
except ImportError:
    # Fallback in case of path differences
    from predict import predict_loan

# Set page configurations
st.set_page_config(
    page_title="FinAI | Loan Eligibility Portal",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Main layout overrides */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #0b0f19;
    color: #f3f4f6;
    background: radial-gradient(circle at 50% 50%, #151d30 0%, #0b0f19 100%);
}

[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}

.main-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

/* Titles */
h1.main-title {
    font-weight: 700;
    font-size: 2.8rem;
    text-align: center;
    background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 50%, #4f46e5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}

p.subtitle {
    text-align: center;
    font-size: 1.1rem;
    color: #9ca3af;
    margin-bottom: 40px;
}

/* Glassmorphism Cards */
.glass-card {
    background: rgba(21, 30, 48, 0.45);
    border-radius: 16px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 30px;
    margin-bottom: 25px;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
}

.glass-card:hover {
    border-color: rgba(99, 102, 241, 0.3);
    box-shadow: 0 12px 40px rgba(99, 102, 241, 0.12);
    transform: translateY(-2px);
}

/* Headers inside cards */
.card-header {
    font-size: 1.4rem;
    font-weight: 600;
    color: #a5b4fc;
    margin-bottom: 20px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    padding-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Results containers */
.result-approved {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.2) 100%);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34d399;
    border-radius: 16px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(16, 185, 129, 0.1);
    animation: fadeIn 0.6s ease;
}

.result-rejected {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.2) 100%);
    border: 1px solid rgba(239, 68, 68, 0.4);
    color: #f87171;
    border-radius: 16px;
    padding: 30px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(239, 68, 68, 0.1);
    animation: fadeIn 0.6s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.result-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 10px;
}

.score-badge {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 1.1rem;
    background: rgba(255, 255, 255, 0.1);
    margin-top: 15px;
}

/* Custom styled parameters list */
.param-row {
    display: flex;
    justify-content: space-between;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.param-row:last-child {
    border-bottom: none;
}

.param-label {
    color: #9ca3af;
}

.param-value {
    font-weight: 600;
    color: #f3f4f6;
}

/* Streamlit input styling customization */
div[data-baseweb="select"] > div, 
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"] > input {
    background-color: rgba(30, 41, 59, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #f3f4f6 !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"]:hover > div, 
div[data-baseweb="input"]:hover > div {
    border-color: rgba(99, 102, 241, 0.5) !important;
}

.stSlider > div > div > div {
    background-color: #6366f1 !important;
}

/* Submit Button styling */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #4338ca 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 28px !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.35) !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #4f46e5 0%, #3730a3 100%) !important;
    box-shadow: 0 6px 25px rgba(99, 102, 241, 0.55) !important;
    transform: translateY(-1px) !important;
}
</style>
""", unsafe_allow_html=True)

# Main Application Layout
st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.markdown("<h1 class='main-title'>💳 FinAI Loan Decision System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Instant Loan Eligibility Evaluation powered by Machine Learning</p>", unsafe_allow_html=True)

# Main form
with st.form("loan_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='card-header'>
            👤 Applicant Information
        </div>
        """, unsafe_allow_html=True)
        
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Marital Status", ["Yes", "No"], help="Is the applicant married?")
        dependents = st.selectbox("Number of Dependents", [0.0, 1.0, 2.0, 3.0], format_func=lambda x: "0" if x == 0.0 else "1" if x == 1.0 else "2" if x == 2.0 else "3+")
        education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed?", ["Yes", "No"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"], help="Area where the property is located.")

    with col2:
        st.markdown("""
        <div class='card-header'>
            💰 Financial Details
        </div>
        """, unsafe_allow_html=True)
        
        applicant_income = st.number_input("Applicant Monthly Income ($)", min_value=0, value=5000, step=500, help="Monthly income of primary applicant.")
        coapplicant_income = st.number_input("Co-Applicant Monthly Income ($)", min_value=0, value=0, step=500, help="Monthly income of the co-applicant.")
        loan_amount = st.number_input("Loan Amount (in Thousands $)", min_value=1, value=150, step=10, help="e.g. 150 means $150,000")
        loan_term = st.selectbox("Loan Term (Months)", [360.0, 240.0, 180.0, 120.0, 84.0, 60.0, 36.0, 12.0], index=0)
        credit_history = st.selectbox("Credit History", [1.0, 0.0], format_func=lambda x: "Good / Cleared Debts" if x == 1.0 else "Poor / Outstanding Debts", help="Repayment history of previous loans.")
        model_type = st.selectbox("Prediction Model", ["Random Forest", "Logistic Regression"], index=0, help="Machine Learning classifier to use.")

    st.markdown("<div style='margin-top: 25px;'>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Evaluate Loan Application")
    st.markdown("</div>", unsafe_allow_html=True)

# Prediction execution
if submitted:
    # 1. Compile input data
    input_data = {
        'Gender': gender,
        'Married': married,
        'Dependents': float(dependents),
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': int(applicant_income),
        'CoapplicantIncome': float(coapplicant_income),
        'LoanAmount': float(loan_amount),
        'Loan_Amount_Term': float(loan_term),
        'Credit_History': float(credit_history),
        'Property_Area': property_area
    }
    
    model_key = 'random_forest' if model_type == "Random Forest" else 'logistic_regression'
    
    with st.spinner("Analyzing credit profile and calculating risk..."):
        try:
            result = predict_loan(input_data, model_key)
            
            # Display Result
            st.markdown("---")
            st.markdown("<h2 style='text-align: center; margin-bottom: 25px;'>Analysis Results</h2>", unsafe_allow_html=True)
            
            res_col1, res_col2 = st.columns([1, 1])
            
            with res_col1:
                # Layout based on Approved or Rejected
                if result['status'] == 'Approved':
                    st.markdown(f"""
                    <div class='result-approved'>
                        <div class='result-title'>🎉 Application Approved</div>
                        <p>Based on the inputs provided, the applicant satisfies the eligibility criteria.</p>
                        <div class='score-badge'>Confidence Score: {result['probability']*100:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='result-rejected'>
                        <div class='result-title'>❌ Application Rejected</div>
                        <p>Based on the inputs provided, the applicant does not satisfy the eligibility criteria.</p>
                        <div class='score-badge'>Confidence Score: {(1 - result['probability'])*100:.1f}% (Rejection)</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with res_col2:
                # Key Indicators Table
                st.markdown(f"""
                <div class='glass-card'>
                    <div class='card-header'>📊 Financial Indicators</div>
                    <div class='param-row'>
                        <span class='param-label'>Total Monthly Income:</span>
                        <span class='param-value'>${applicant_income + coapplicant_income:,.2f}</span>
                    </div>
                    <div class='param-row'>
                        <span class='param-label'>Requested Loan:</span>
                        <span class='param-value'>${loan_amount * 1000:,.2f}</span>
                    </div>
                    <div class='param-row'>
                        <span class='param-label'>Debt-to-Income Ratio:</span>
                        <span class='param-value'>{(loan_amount * 1000) / (applicant_income + coapplicant_income + 1):.2f}</span>
                    </div>
                    <div class='param-row'>
                        <span class='param-label'>Credit History Check:</span>
                        <span class='param-value' style='color: {"#34d399" if credit_history == 1.0 else "#f87171"}'>
                            {"PASS" if credit_history == 1.0 else "FAIL"}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Dynamic visual guidance
                if credit_history == 0.0:
                    st.warning("⚠️ **Warning:** The credit history record is flagged. Applicants with outstanding debts or poor repayment history have a significantly higher risk score.")
                if (applicant_income + coapplicant_income) < 3000:
                    st.warning("⚠️ **Warning:** Low total household income relative to loan requirements. Consider adding a co-applicant or requesting a smaller loan amount.")
                    
        except FileNotFoundError:
            st.error("🚨 **Error:** Trained models or preprocessing encoders could not be found. Please run the training pipeline first: `python src/train.py`.")
        except Exception as e:
            st.error(f"🚨 **An unexpected error occurred:** {e}")

st.markdown("</div>", unsafe_allow_html=True)
