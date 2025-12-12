import streamlit as st
import joblib

# Title and Header
st.markdown("<h1 style='text-align: center; color: red;'>LOAN PPROVE DATA ANALYSIS</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: blue;'>Data Analysis</h2>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: blue;'>By Suhail TP</h4>", unsafe_allow_html=True)
st.title("PREDICTION ANALYSIS")

# User Inputs
Recent_payment_time = st.number_input("Enter your recent payment time(Eg:123):")
time_since_first_deliquency = st.number_input("Time since first deliquency")
time_since_recent_deliquency = st.number_input("Time since recent deliquency")
num_times_delinquent = st.number_input("Num times delinquent")
max_delinquency_level = st.number_input("Max delinquency level")
max_recent_level_of_deliq = st.number_input("Max recent level of deliq")
num_deliq_6mts = st.number_input("Num deliq 6mts(1-12)")
num_deliq_12mts = st.number_input("Num deliq 12mts(1-28)")
num_deliq_6_12mts = st.number_input("Num deliq 6_12mts(1-17)")
max_deliq_6mts = st.number_input("Max deliq 6mts")
max_deliq_12mts = st.number_input("Max deliq 12mts")
num_times_30p_dpd = st.number_input("Num times 30p dpd")
num_times_60p_dpd = st.number_input("Num times 60p dpd")
num_std = st.number_input("Num std")
num_std_6mts = st.number_input("Num std 6mts(1-60)")
num_std_12mts = st.number_input("Num std 12mts")
num_sub = st.number_input("Num sub")
num_sub_6mts = st.number_input("Num sub 6mts")
num_sub_12mts = st.number_input("Num sub 12mts")
num_dbt = st.number_input("Num dbt")
num_dbt_6mts = st.number_input("Num dbt 6mts")
num_dbt_12mts = st.number_input("Num dbt 12mts")
num_lss = st.number_input("Num lss:")
num_lss_6mts = st.number_input("Num lss 6mts:")
num_lss_12mts = st.number_input("Num lss 12mts:")
recent_level_of_deliq = st.number_input("Recent level of deliq:")
tot_enq = st.number_input("Tot enq:")
CC_enq = st.number_input("CC enq:")
CC_enq_L6m = st.number_input("CC enq L6m:")
CC_enq_L12m = st.number_input("CC enq L12m:")
PL_enq = st.number_input("PL enq")
PL_enq_L6m = st.number_input("PL enq L6m")
PL_enq_L12m = st.number_input("PL enq L12m")
time_since_recent_enq = st.number_input("Time since recent enq")
enq_L12m = st.number_input("Enq L12m:")
enq_L6m = st.number_input("Enq L6m:")
enq_L3m = st.number_input("Enq L3m:")
AGE = st.number_input("Enter your Age")
NETMONTHLYINCOME = st.number_input("Enter your Monthly income:")
Time_With_Curr_Empr = st.number_input("Time With Curr Empr:")
pct_of_active_TLs_ever = st.number_input("Pct of active TLs ever:")
pct_opened_TLs_L6m_of_L12m = st.number_input("Pct opened TLs L6m of L12m:")
pct_currentBal_all_TL = st.number_input("Pct currentBal all TL:")
CC_utilization = st.number_input("CC utilization:")
CC_Flag = st.number_input("CC Flag (0 or 1):")
PL_utilization = st.number_input("PL utilization:")
PL_Flag = st.number_input("PL Flag (0 or 1):")
pct_PL_enq_L6m_of_L12m = st.number_input("Pct PL enq L6m of L12m:")
pct_CC_enq_L6m_of_L12m = st.number_input("Pct CC enq L6m of L12m:")
pct_PL_enq_L6m_of_ever = st.number_input("Pct PL enq L6m of ever:")
pct_CC_enq_L6m_of_ever = st.number_input("pct CC_enq L6m_of ever:")
max_unsec_exposure_inPct = st.number_input("Max unsec exposure inPct:")
HL_Flag = st.number_input("HL Flag (0 or 1):")
GL_Flag = st.number_input("GL Flag (0 or 1):")
Credit_Score = st.number_input("Credit Score:")

MARITALSTATUS = st.radio("Select your martial status:", options=["Single", "Married"])
GENDER = st.radio("Select your sex:", options=["M", "F"])
EDUCATION = st.selectbox("Education:", options=['SSC','12TH','UNDER GRADUATE','GRADUATE','POST-GRADUATE','PROFESSIONAL','OTHERS'])
last_prod_enq2 = st.selectbox("Select your last prod enq2:", options=['PL','ConsumerLoan','others','AL','CC','HL'])
first_prod_enq2 = st.selectbox("Select your first prod enq2:", options=['PL','ConsumerLoan','others','AL','HL','CC'])

#load pipeline
pipeline = joblib.load("/Users/mac/loan_pipeline.pkl")

edu_map = pipeline["edu_map"]
encoders = pipeline["encoders"]
scaler = pipeline["scaler"]
model = pipeline["model"]

# Encoding
education = edu_map[EDUCATION]

# Encode categorical variables
gender = encoders["GENDER"].transform([GENDER])[0]
marital = encoders["MARITALSTATUS"].transform([MARITALSTATUS])[0]
last_prod = encoders["last_prod_enq2"].transform([last_prod_enq2])[0]
first_prod = encoders["first_prod_enq2"].transform([first_prod_enq2])[0]

# Prediction
if st.button("Predict"):
    input_data=[[Recent_payment_time,
                 time_since_first_deliquency,
                 time_since_recent_deliquency,
                 num_times_delinquent,
                 max_delinquency_level,
                 max_recent_level_of_deliq,
                 num_deliq_6mts,
                 num_deliq_12mts,
                 num_deliq_6_12mts,
                 max_deliq_6mts,
                 max_deliq_12mts,
                 num_times_30p_dpd,
                 num_times_60p_dpd,
                 num_std,
                 num_std_6mts,
                 num_std_12mts,
                 num_sub,
                 num_sub_6mts,
                 num_sub_12mts,
                 num_dbt,
                 num_dbt_6mts,
                 num_dbt_12mts,
                 num_lss,
                 num_lss_6mts,
                 num_lss_12mts,
                 recent_level_of_deliq,
                 tot_enq,
                 CC_enq,
                 CC_enq_L6m,
                 CC_enq_L12m,
                 PL_enq,
                 PL_enq_L6m,
                 PL_enq_L12m,
                 time_since_recent_enq,
                 enq_L12m,
                 enq_L6m,
                 enq_L3m,
                 marital,
                 education,
                 AGE,
                 gender,
                 NETMONTHLYINCOME,
                 Time_With_Curr_Empr,
                 pct_of_active_TLs_ever,
                 pct_opened_TLs_L6m_of_L12m,
                 pct_currentBal_all_TL,
                 CC_utilization,
                 CC_Flag,
                 PL_utilization,
                 PL_Flag,
                 pct_PL_enq_L6m_of_L12m,
                 pct_CC_enq_L6m_of_L12m,
                 pct_PL_enq_L6m_of_ever,
                 pct_CC_enq_L6m_of_ever,
                 max_unsec_exposure_inPct,
                 HL_Flag,
                 GL_Flag,
                 last_prod,
                 first_prod,
                 Credit_Score]]
    result = model.predict(scaler.transform(input_data))[0]
    if result == 0:
        st.error("❌ **Loan Not Approved**\n\n"
        "Applicant meets only Phase-1 criteria.\n"
        "👉 Visit the nearest branch for further evaluation.")
    elif result == 1:
        st.error("❌ **Loan Not Approved**\n\n"
        "Applicant may clear Phase-1 & Phase-2 only.\n"
        "👉 Additional documents might be required. Please visit the branch.")
    elif result == 2:
        st.error("❌ **Loan Not Approved**\n\n"
        "Applicant may clear up to Phase-3, but not final approval.\n"
        "👉 Consult a loan officer at your nearest branch.")
    else:
        st.success("✔️ **Loan Approved**\n\n"
        "Applicant is eligible for all approval phases (P1–P4).\n"
        "👉 Visit the branch to complete final paperwork.")

    



    
        
        

        
    