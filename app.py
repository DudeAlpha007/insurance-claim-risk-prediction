# Insurance Claim Prediction Streamlit App

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Insurance Claim Prediction System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS 
# ============================================

st.markdown(
    """
    <style>
    /* HIDE DEFAULT HEADER, FOOTER */
    #MainMenu {visibility: hidden;}
    /* header {visibility: hidden;} */
    footer {visibility: hidden;}
    
    /* HIDE SCROLLBARS ONLY ON THE MAIN PAGE */
    section[data-testid="stMain"] ::-webkit-scrollbar {
        width: 0px;
        background: transparent;
    }

    /* ELIMINATE PADDING TO FIT EVERYTHING IN ONE FRAME */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 0rem !important;
        max-width: 95% !important;
    }
    
    /* REMOVE EXTRA SPACE AT THE TOP OF THE SIDEBAR */
    [data-testid="stSidebarUserContent"] {
        padding-top: 0rem !important;
    }

    /* COMPACT TITLES */
    .title {
        font-size: 42px;
        font-weight: bold;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 10px;
    }

    /* SLIDE 1 IMAGE SCALING - Fit to Screen */
    [data-testid="stImage"] img {
        max-height: 85vh; 
        width: 100%;
        object-fit: contain;
    }

    /* SLIDE 3 & 4: RESTORED ORIGINAL RESULT CARDS */
    .result-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 18px;
        padding: 25px;
        height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        margin-bottom: 20px;
    }

    .result-title {
        font-size: 26px;
        font-weight: 600;
        color: #FFFFFF;
        margin-bottom: 20px;
    }

    .result-value {
        font-size: 38px;
        font-weight: bold;
        color: #FFFFFF;
        text-align: center;
    }

    .result-risk {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
    }

    /* RESTORED INFO BOXES */
    .info-box, .info-box-left {
        background-color: #161B22;
        padding: 30px;
        border-radius: 18px;
        border: 1px solid #30363D;
        height: 240px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .info-box-left {
        align-items: center; 
        text-align: center; 
    }
    
    .info-title {
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 15px;
        line-height: 1.2;
    }
    
    .info-subtitle {
        font-size: 20px;
        font-weight: bold;
        color: #FFFFFF;
        margin-bottom: 12px;
    }
    
    .info-desc {
        font-size: 18px;
        color: #CCCCCC;
        line-height: 1.5;
    }

    .risk-high { color: #FF4B4B; }
    .risk-medium { color: #FFA500; }
    .risk-low { color: #00C853; }

    /* BUTTONS */
    section[data-testid="stMain"] div[data-testid="stButton"] {
        margin: 0 !important; padding: 0 !important; height: 0 !important; width: 0 !important;
    }

    section[data-testid="stMain"] button[kind="primary"], 
    section[data-testid="stMain"] button[kind="secondary"] {
        position: fixed !important;
        bottom: 30px !important;
        width: 55px !important; 
        height: 55px !important;
        border-radius: 12px !important;
        padding: 0 !important; 
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        font-size: 24px !important;
        z-index: 9999 !important;
        border: none !important;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.4) !important;
        transition: transform 0.2s, background-color 0.2s !important;
        overflow: visible !important;
    }

    section[data-testid="stMain"] button[kind="primary"] { right: 30px !important; background-color: #FF4B4B !important; color: white !important;}
    section[data-testid="stMain"] button[kind="primary"]:hover { background-color: #ff3333 !important; transform: scale(1.05) !important; }
    
    section[data-testid="stMain"] button[kind="secondary"] { left: 30px !important; background-color: #30363D !important; color: white !important;}
    section[data-testid="stMain"] button[kind="secondary"]:hover { background-color: #4A535C !important; transform: scale(1.05) !important; }

    section[data-testid="stMain"] button[kind="primary"]::after, 
    section[data-testid="stMain"] button[kind="secondary"]::after {
        position: absolute;
        bottom: 70px;
        background-color: #E8EAED; 
        color: #202124; 
        padding: 8px 14px;
        border-radius: 6px;
        font-size: 15px;
        font-weight: 500;
        opacity: 0;
        visibility: hidden;
        transition: all 0.2s ease-in-out;
        white-space: nowrap;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        pointer-events: none;
        content: attr(data-tooltip); 
    }

    section[data-testid="stMain"] button[kind="primary"]:hover::after, 
    section[data-testid="stMain"] button[kind="secondary"]:hover::after {
        opacity: 1;
        visibility: visible;
        bottom: 65px; 
    }

    section[data-testid="stMain"] button[kind="primary"]::after { right: 0; }
    section[data-testid="stMain"] button[kind="secondary"]::after { left: 0; }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================
# TOOLTIP HELP TEXTS
# ============================

subscription_length_help = """
Total duration of the active policy in years.

Historically, customers with very short subscription histories tend to exhibit
less predictable driving behavior, slightly elevating claim risk.
"""

vehicle_age_help = """
Age of the insured vehicle in years.

Older vehicles generally lack modern advanced safety features and often correlate
with a higher frequency of maintenance-related claims.
"""

customer_age_help = """
Age of the primary policyholder.

Statistically, demographic risk curves show that very young drivers and senior
drivers experience different claim probabilities compared to middle-aged drivers.
"""

traffic_area_help = """
Population and traffic density category of the customer's primary driving region.

High-traffic, dense urban areas significantly increase the daily probability of
minor collisions and fender-benders.
"""

segment_help = """
The automotive market classification of the vehicle:

* **A** : Small Hatchback (Nimble, city driving)
* **B1/B2** : Sedans & Premium Hatchbacks (Balanced utility)
* **C1/C2** : SUVs (Larger mass, handling dynamics, higher repair costs)
"""

fuel_type_help = """
The primary combustion fuel source of the vehicle.

Variations in engine build (like high-pressure CNG tanks or heavy Diesel blocks)
can subtly influence repair costs and overall operational risk.
"""

airbags_help = """
Total number of airbags deployed in the vehicle cabin.

A critical passive safety metric. Vehicles with multiple airbags significantly
mitigate passenger injury severity, directly reducing medical claim payouts.
"""

displacement_help = """
Total volume of all engine cylinders (measured in CC).

Larger displacements dictate a more powerful, heavier engine block,
altering the vehicle's speed capability and handling dynamics.
"""

cylinder_help = """
Number of combustion cylinders in the engine block.

This directly affects power delivery smoothness, engine weight,
and overall driving stability.
"""

turning_radius_help = """
The minimum circular space required for the vehicle to complete a U-turn (in meters).

Larger turning radii make vehicles harder to maneuver in tight urban environments
or parking lots, raising the risk of scrapes and bumps.
"""

gross_weight_help = """
The maximum certified operating weight of the vehicle (in kg).

Heavier vehicles carry more kinetic energy, requiring significantly longer braking
distances which can increase rear-end collision probability.
"""

ncap_rating_help = """
Global standard safety rating (0 to 5 stars) based on rigorous crash testing.

Higher NCAP scores confirm superior structural integrity,
drastically lowering the risk of total loss claims.
"""

total_safety_features_help = """
Aggregate count of active Advanced Driver Assistance Systems (ADAS)
like ABS, ESC, parking sensors, and brake assist.

More active features help prevent accidents before they happen.
"""

torque_value_help = """
The rotational pulling force of the engine (measured in Nm).

High torque allows rapid acceleration, which may correlate with
more aggressive driving patterns and a higher accident risk.
"""

torque_rpm_help = """
The engine speed (in RPM) at which peak torque is delivered.

Lower RPM torque means quicker acceleration at low speeds,
potentially increasing collision likelihood in traffic.
"""

power_value_help = """
The peak horsepower output of the engine.

High-horsepower vehicles have higher top speeds and are often
associated with sportier driving behavior and higher claim severity.
"""

power_rpm_help = """
The engine speed (in RPM) required to reach maximum horsepower.

Higher RPM power delivery can encourage aggressive acceleration behavior.
"""

# ============================================
# LOAD MODEL 
# ============================================

try:
    with open("Insurance Claim Prediction Model.pkl", "rb") as f:
        loaded_model = pickle.load(f)

    with open("Insurance Threshold.pkl", "rb") as f:
        loaded_threshold = pickle.load(f)
except Exception as e:
    st.error(f"Model Loading Error: {e}")
    loaded_model = None

# ============================================
# SESSION STATE INITIALIZATION 
# ============================================

if 'current_step' not in st.session_state:
    st.session_state.current_step = 1

# ============================================
# SLIDE 1: LANDING PAGE
# ============================================
if st.session_state.current_step == 1:
    
    st.markdown('<style>section[data-testid="stMain"] button[kind="primary"]::after { content: "Enter System"; }</style>', unsafe_allow_html=True)
    
    try:
        st.image("image.png", use_container_width=True)
    except:
        pass

    if st.button("➡️", type="primary"):
        st.session_state.current_step = 2
        st.rerun()

# ============================================
# SLIDE 2: INPUTS PAGE
# ============================================
elif st.session_state.current_step == 2:

    if 'demo_case' not in st.session_state:
        st.session_state.demo_case = "Custom Input"

    st.sidebar.title("📌 Model Information")

    st.sidebar.info(
        """
        Final Model : Tuned Balanced XGBoost
        
        Final Threshold : 0.55
        
        ROC-AUC : 0.679
        
        F1 Score : 0.188
        """
    )
    
    st.sidebar.markdown("### 🎯 Selected Profile")
    st.sidebar.success(f"Active Profile: {st.session_state.demo_case}")

    st.sidebar.markdown("## 🚘 Demo Risk Profiles")
    
    if st.sidebar.button("🧑 Custom Input", use_container_width=True):
        st.session_state.demo_case = "Custom Input"
        st.rerun()

    if st.sidebar.button("🟢 Good Driver", use_container_width=True):
        st.session_state.demo_case = "Good Driver"
        st.rerun()

    if st.sidebar.button("🟡 Moderate Driver", use_container_width=True):
        st.session_state.demo_case = "Moderate Driver"
        st.rerun()

    if st.sidebar.button("🔴 Risky Driver", use_container_width=True):
        st.session_state.demo_case = "Risky Driver"
        st.rerun()

    demo_case = st.session_state.demo_case

    # ---- DEFAULT VALUES ----
    subscription_length_default = 5.0
    vehicle_age_default = 2.0
    customer_age_default = 35
    traffic_area_default = "Low Traffic Area"
    segment_default = "B1"
    fuel_default = "Petrol"

    airbags_default = 2
    displacement_default = 1200
    cylinder_default = 4
    turning_radius_default = 4.8
    gross_weight_default = 1300
    ncap_default = 3

    safety_default = 4
    torque_default = 110
    torque_rpm_default = 3500
    power_default = 90.0
    power_rpm_default = 5000

# Adjusted Demo Profile Values to trigger correct risk buckets
    if demo_case == "Good Driver":
        subscription_length_default, vehicle_age_default, customer_age_default = 4.2, 14.2, 19
        traffic_area_default = "High Traffic Area"
        segment_default, fuel_default = "B2", "CNG"
        airbags_default, displacement_default, cylinder_default = 0, 1579, 3
        turning_radius_default, gross_weight_default, ncap_default = 5.3, 1292, 3
        safety_default, torque_default, torque_rpm_default = 0, 286, 4162
        power_default, power_rpm_default = 130.6, 1866

    elif demo_case == "Moderate Driver":
        subscription_length_default, vehicle_age_default, customer_age_default = 4.0, 3.0, 32
        traffic_area_default = "Medium Traffic Area"
        segment_default, fuel_default = "B1", "Diesel"
        airbags_default, displacement_default, cylinder_default = 2, 1500, 4
        turning_radius_default, gross_weight_default, ncap_default = 4.8, 1400, 3
        safety_default, torque_default, torque_rpm_default = 4, 180, 3500
        power_default, power_rpm_default = 100.0, 4500

    elif demo_case == "Risky Driver":
        subscription_length_default, vehicle_age_default, customer_age_default = 0.5, 0.0, 22
        traffic_area_default = "High Traffic Area"
        segment_default, fuel_default = "A", "CNG"
        airbags_default, displacement_default, cylinder_default = 0, 800, 3
        turning_radius_default, gross_weight_default, ncap_default = 5.2, 1000, 0
        safety_default, torque_default, torque_rpm_default = 0, 80, 5000
        power_default, power_rpm_default = 60.0, 6500
        
    elif demo_case == "Custom Input" and "saved_inputs" in st.session_state:
        saved = st.session_state.saved_inputs
        subscription_length_default = saved.get('subscription_length', 5.0)
        vehicle_age_default = saved.get('vehicle_age', 2.0)
        customer_age_default = saved.get('customer_age', 35)
        traffic_area_default = saved.get('traffic_area', "Low Traffic Area")
        segment_default = saved.get('segment', "B1")
        fuel_default = saved.get('fuel_type', "Petrol")
        airbags_default = saved.get('airbags', 2)
        displacement_default = saved.get('displacement', 1200)
        cylinder_default = saved.get('cylinder', 4)
        turning_radius_default = saved.get('turning_radius', 4.8)
        gross_weight_default = saved.get('gross_weight', 1300)
        ncap_default = saved.get('ncap_rating', 3)
        safety_default = saved.get('total_safety_features', 4)
        torque_default = saved.get('torque_value', 110)
        torque_rpm_default = saved.get('torque_rpm', 3500)
        power_default = saved.get('power_value', 90.0)
        power_rpm_default = saved.get('power_rpm', 5000)
    
    st.markdown('''<style>
        section[data-testid="stMain"] button[kind="primary"]::after { content: "Predict Risk"; }
        section[data-testid="stMain"] button[kind="secondary"]::after { content: "Go Back"; }
    </style>''', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 👤 Customer Details")
        subscription_length = st.slider("Subscription Length", 0.0, 15.0, float(subscription_length_default), help=subscription_length_help)
        vehicle_age = st.slider("Vehicle Age", 0.0, 20.0, float(vehicle_age_default), help=vehicle_age_help)
        customer_age = st.slider("Customer Age", 18, 80, int(customer_age_default), help=customer_age_help)
        traffic_area = st.selectbox("Traffic Density Area", ["Low Traffic Area", "Medium Traffic Area", "High Traffic Area"], index=["Low Traffic Area", "Medium Traffic Area", "High Traffic Area"].index(traffic_area_default), help=traffic_area_help)
        segment = st.selectbox("Vehicle Segment", ["A", "B1", "B2", "C1", "C2"], index=["A", "B1", "B2", "C1", "C2"].index(segment_default), help=segment_help)
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"], index=["Petrol", "Diesel", "CNG"].index(fuel_default), help=fuel_type_help)

        if traffic_area == "Low Traffic Area": region_density = 5000
        elif traffic_area == "Medium Traffic Area": region_density = 30000
        else: region_density = 70000

    with col2:
        st.markdown("### 🚗 Vehicle Details")
        airbags = st.slider("Airbags", 0, 6, int(airbags_default), help=airbags_help)
        displacement = st.number_input("Engine Displacement", min_value=700, max_value=2000, value=int(displacement_default), help=displacement_help)
        
        if fuel_type == "CNG":
            cylinder = st.selectbox("Cylinder", [3, 4], index=[3, 4].index(cylinder_default) if cylinder_default in [3, 4] else 0, help=cylinder_help)
        else:
            cylinder = cylinder_default
            
        turning_radius = st.slider("Turning Radius", 4.0, 6.0, float(turning_radius_default), help=turning_radius_help)
        gross_weight = st.number_input("Gross Weight", min_value=900, max_value=2500, value=int(gross_weight_default), help=gross_weight_help)
        ncap_rating = st.slider("NCAP Rating", 0, 5, int(ncap_default), help=ncap_rating_help)

    with col3:
        st.markdown("### 🛡️ Safety & Performance")
        total_safety_features = st.slider("Total Safety Features", 0, 8, int(safety_default), help=total_safety_features_help)
        torque_value = st.number_input("Torque Value", min_value=40, max_value=300, value=int(torque_default), help=torque_value_help)
        torque_rpm = st.number_input("Torque RPM", min_value=1000, max_value=7000, value=int(torque_rpm_default), help=torque_rpm_help)
        power_value = st.number_input("Power Value", min_value=20.0, max_value=200.0, value=float(power_default), help=power_value_help)
        power_rpm = st.number_input("Power RPM", min_value=1000, max_value=8000, value=int(power_rpm_default), help=power_rpm_help)

    if st.button("⬅️", type="secondary"):
        st.session_state.current_step = 1
        st.rerun()

    if st.button("🔍", type="primary"):
        st.session_state.demo_case = "Custom Input"
        st.session_state.saved_inputs = {
            'subscription_length': subscription_length, 'vehicle_age': vehicle_age, 'customer_age': customer_age,
            'traffic_area': traffic_area, 'segment': segment, 'fuel_type': fuel_type, 'airbags': airbags,
            'displacement': displacement, 'cylinder': cylinder, 'turning_radius': turning_radius,
            'gross_weight': gross_weight, 'ncap_rating': ncap_rating, 'total_safety_features': total_safety_features,
            'torque_value': torque_value, 'torque_rpm': torque_rpm, 'power_value': power_value, 'power_rpm': power_rpm
        }

        power_to_weight = power_value / gross_weight
        vehicle_size = 4000 * 1700

        # =======================================================
        # 🚨 NEW FEATURE ENGINEERING
        # =======================================================
        # Note: Ensure these formulas match what you did in Jupyter!
        vehicle_risk_score = float(vehicle_age) / (int(ncap_rating) + 1.0) 
        vehicle_usage_risk = float(vehicle_age) * float(subscription_length)
        engine_stress_index = float(power_rpm) / float(torque_rpm) if torque_rpm > 0 else 1.0
        vehicle_density_risk = float(region_density) / 1000.0

        if vehicle_age <= 2: vehicle_age_group = "New"
        elif vehicle_age <= 5: vehicle_age_group = "Moderate"
        elif vehicle_age <= 10: vehicle_age_group = "Old"
        else: vehicle_age_group = "Very_Old"

        if customer_age <= 40: customer_age_group = "Young"
        elif customer_age <= 55: customer_age_group = "Middle_Age"
        else: customer_age_group = "Senior"

        if power_value <= 70: power_category = "Low"
        elif power_value <= 100: power_category = "Medium"
        else: power_category = "High"

        # Construct DataFrame EXACTLY matching the expected features
        input_data = pd.DataFrame({
            'subscription_length': [subscription_length], 'vehicle_age': [vehicle_age], 'customer_age': [customer_age],
            'region_code': ['C18'], 'region_density': [region_density], 'segment': [segment], 'model': ['M1'],
            'fuel_type': [fuel_type], 'engine_type': ['F8D Petrol Engine'], 'airbags': [airbags],
            'is_esc': [1 if total_safety_features >= 4 else 0], 'is_adjustable_steering': [1 if total_safety_features >= 3 else 0],
            'is_tpms': [1 if total_safety_features >= 4 else 0], 'is_parking_sensors': [1 if total_safety_features >= 2 else 0],
            'is_parking_camera': [1 if total_safety_features >= 5 else 0], 'rear_brakes_type': ['Drum'],
            'displacement': [displacement], 'cylinder': [cylinder], 'transmission_type': ['Manual'],
            'steering_type': ['Power'], 'turning_radius': [turning_radius], 'length': [4000],
            'width': [1700], 'gross_weight': [gross_weight], 'is_front_fog_lights': [1 if total_safety_features >= 3 else 0],
            'is_rear_window_wiper': [1 if total_safety_features >= 4 else 0], 'is_rear_window_washer': [1 if total_safety_features >= 4 else 0],
            'is_rear_window_defogger': [1 if total_safety_features >= 4 else 0], 'is_brake_assist': [1 if total_safety_features >= 5 else 0],
            'is_power_door_locks': [1 if total_safety_features >= 3 else 0], 'is_central_locking': [1 if total_safety_features >= 3 else 0],
            'is_power_steering': [1 if total_safety_features >= 2 else 0], 'is_driver_seat_height_adjustable': [1 if total_safety_features >= 4 else 0],
            'is_day_night_rear_view_mirror': [1 if total_safety_features >= 4 else 0], 'is_ecw': [1 if total_safety_features >= 5 else 0],
            'is_speed_alert': [1 if total_safety_features >= 2 else 0], 'ncap_rating': [ncap_rating],
            'torque_value': [torque_value], 'torque_rpm': [torque_rpm], 'power_value': [power_value],
            'power_rpm': [power_rpm], 'total_safety_features': [total_safety_features],
            'vehicle_age_group': [vehicle_age_group], 'customer_age_group': [customer_age_group],
            'power_category': [power_category], 'power_to_weight': [power_to_weight], 'vehicle_size': [vehicle_size],
            
            # 🚨 NEW COLUMNS HERE:
            'vehicle_risk_score': [vehicle_risk_score],
            'vehicle_usage_risk': [vehicle_usage_risk],
            'engine_stress_index': [engine_stress_index],
            'vehicle_density_risk': [vehicle_density_risk]
        })
            
        st.session_state.input_data = input_data

        if loaded_model is None:
            st.error("🚨 Critical Error: The Model (.pkl) files were not found! Please check your file paths.")
            st.stop()

        try:
            # Attempt to predict
            probability = loaded_model.predict_proba(input_data)[:,1][0]
            thresh = loaded_threshold if loaded_threshold is not None else 0.55
            prediction = int(probability >= thresh)
            
        except Exception as e:
            # IF PREDICTION FAILS, SHOW THE REAL ERROR SO WE CAN FIX IT
            st.error(f"🚨 Model Prediction Failed! Error details:\n{str(e)}")
            st.stop()

        if probability >= 0.55:
            risk_level, risk_class, driver_behavior = "HIGH RISK", "risk-high", "Risky Driver"
            behavior_text = "Driver profile shows elevated insurance risk patterns. Higher probability of claim occurrence based on vehicle and customer characteristics."
            st.session_state.show_balloons = False
            
        elif probability >= 0.40:  # 🚨 Updated from 0.3 to 0.40
            risk_level, risk_class, driver_behavior = "MEDIUM RISK", "risk-medium", "Moderate Driver"
            behavior_text = "Driver profile shows balanced insurance behavior with moderate claim probability."
            st.session_state.show_balloons = False
            
        else: # Now 34.38% will naturally fall into the Safe Driver bucket!
            risk_level, risk_class, driver_behavior = "LOW RISK", "risk-low", "Safe Driver"
            behavior_text = "Driver profile appears stable with a low predicted claim probability."
            st.session_state.show_balloons = True

        st.session_state.res_prob, st.session_state.res_pred = probability, prediction
        st.session_state.res_risk_level, st.session_state.res_risk_class = risk_level, risk_class
        st.session_state.res_driver_behavior, st.session_state.res_behavior_text = driver_behavior, behavior_text

        st.session_state.current_step = 3
        st.rerun()

# ============================================
# SLIDE 3: RESULTS PAGE
# ============================================
elif st.session_state.current_step == 3:
    
    if st.session_state.get('show_balloons', False):
        st.balloons()
        st.session_state.show_balloons = False 
        
    st.markdown('<style>section[data-testid="stMain"] button[kind="secondary"]::after { content: "Edit Values"; } section[data-testid="stMain"] button[kind="primary"]::after { content: "AI Explanation"; }</style>', unsafe_allow_html=True)

    st.markdown('<div class="title" style="margin-bottom:30px;">📊 Prediction Results</div>', unsafe_allow_html=True)

    prob, pred = st.session_state.res_prob, st.session_state.res_pred
    risk_level, risk_class = st.session_state.res_risk_level, st.session_state.res_risk_class
    driver_behavior, behavior_text = st.session_state.res_driver_behavior, st.session_state.res_behavior_text

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="result-card"><div class="result-title">Claim Probability</div><div class="result-value">{prob:.2%}</div></div>', unsafe_allow_html=True)
    with m2:
        pred_text = "Claim Likely" if pred == 1 else "Claim Not Likely"
        st.markdown(f'<div class="result-card"><div class="result-title">Prediction</div><div class="result-value">{pred_text}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="result-card"><div class="result-title">Risk Level</div><div class="result-risk {risk_class}">{risk_level}</div></div>', unsafe_allow_html=True)

    i1, i2 = st.columns(2)
    with i1:
        if risk_level == "HIGH RISK":
            msg_title, msg_desc, color = "⚠️ High Claim Risk", "High probability of insurance claim detected.", "#FF4B4B"
        elif risk_level == "MEDIUM RISK":
            msg_title, msg_desc, color = "⚠️ Moderate Claim Risk", "Moderate insurance claim probability detected.", "#FFA500"
        else:
            msg_title, msg_desc, color = "✅ Safe Insurance Profile", "Customer is highly unlikely to raise a claim.", "#00C853"
            
        st.markdown(f"""
            <div class="info-box-left">
                <div class="info-title" style="color:{color};">{msg_title}</div>
                <div class="info-desc">{msg_desc}</div>
            </div>
        """, unsafe_allow_html=True)

    with i2:
        st.markdown(f"""
            <div class="info-box">
                <div class="info-title" style="color:#3EA6FF;">Driver Behavior Analysis</div>
                <div class="info-subtitle">Classification : {driver_behavior}</div>
                <div class="info-desc">{behavior_text}</div>
            </div>
        """, unsafe_allow_html=True)

    if st.button("⬅️", type="secondary"):
        st.session_state.current_step = 2
        st.rerun()

    if st.button("🧠", type="primary"):
        st.session_state.current_step = 4
        st.rerun()

# ============================================
# SLIDE 4: AI DECISION EXPLANATION (WIRE-TAP FIX)
# ============================================
elif st.session_state.current_step == 4:

    st.markdown(
        '''
        <style>
        section[data-testid="stMain"] button[kind="secondary"]::after {
            content: "Back to Results";
        }

        section[data-testid="stMain"] button[kind="primary"] {
            display: none !important;
        }
        </style>
        ''',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="title" style="margin-bottom:10px;">🧠 AI Decision Explanation</div>',
        unsafe_allow_html=True
    )

    prob = float(st.session_state.res_prob)
    risk_level = st.session_state.res_risk_level

    st.markdown(
        f"""
        <h3 style='text-align:center;
                   color:#FFFFFF;
                   margin-bottom:20px;'>

            Step-by-Step Risk Journey:
            <span style='color:#3EA6FF;'>
                {prob:.2%}
            </span>

        </h3>
        """,
        unsafe_allow_html=True
    )

    try:

        import re
        import json
        import shap
        import numpy as np
        import pandas as pd
        import plotly.graph_objects as go

        input_df = st.session_state.input_data.copy()

        # =========================================================
        # 1. EXTRACT MODEL & PREPROCESSOR
        # =========================================================
        if hasattr(loaded_model, "steps"):
            preprocessor = loaded_model[:-1]
            classifier = loaded_model[-1]
            X_transformed = preprocessor.transform(input_df)
            raw_names = preprocessor.get_feature_names_out()
            clean_names = [name.split('__')[-1].replace('_', ' ').title() for name in raw_names]
        else:
            classifier = loaded_model
            X_transformed = input_df
            clean_names = [c.replace('_', ' ').title() for c in input_df.columns]

        # =========================================================
        # 2. THE OFFICIAL XGBOOST/SHAP FIX
        # =========================================================
        booster = classifier.get_booster()
        
        import json
        import re
        
        try:
            # 1. Read the corrupted internal config
            conf = json.loads(booster.save_config())
            raw_score = str(conf['learner']['learner_model_param']['base_score'])
            
            # 2. Extract ONLY the pure math number (e.g. from "[5.0001556E-1]")
            clean_score_match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', raw_score)
            
            if clean_score_match:
                clean_score = float(clean_score_match.group())
                # 3. 🚨 Force XGBoost to overwrite its internal memory with a clean float
                booster.set_param({'base_score': clean_score})
        except Exception:
            booster.set_param({'base_score': 0.5})

        # =========================================================
        # 3. SHAP EXTRACTION 
        # =========================================================
        # Now SHAP will read the clean float and process normally
        explainer = shap.TreeExplainer(booster)
        shap_vals = explainer.shap_values(X_transformed)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        shap_vals = np.array(shap_vals)

        # =========================================================
        # 4. BASE VALUE & SHAP VALUE CLEANING
        # =========================================================
        raw_base = explainer.expected_value
        if isinstance(raw_base, (list, np.ndarray)):
            raw_base = np.array(raw_base).flatten()[-1]

        s_base = str(raw_base).replace('[', '').replace(']', '').replace('"', '').replace("'", "")
        match_base = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s_base)
        base_val_clean = float(match_base[0]) if match_base else 0.0

        current_shap_vals = np.array(shap_vals[0]).flatten()
        
        clean_shap_vals = []
        for x in current_shap_vals:
            s_val = str(x).replace('[', '').replace(']', '').replace('"', '').replace("'", "")
            m = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s_val)
            clean_shap_vals.append(float(m[0]) if m else 0.0)
        clean_shap_vals = np.array(clean_shap_vals)

        # =========================================================
        # 5. MATH & LOGIT FUNCTION
        # =========================================================
        def logit(x): return 1 / (1 + np.exp(-x))

        base_prob_pct = round(logit(base_val_clean) * 100, 1)

        total_prob_movement = (prob * 100) - base_prob_pct

        total_logit_change = np.sum(np.abs(clean_shap_vals))

        if total_logit_change == 0:
            total_logit_change = 1e-9

        # =========================================================
        # 6. FEATURE IMPACTS
        # =========================================================
        impact_list = []

        for i, name in enumerate(clean_names):
            if i < len(clean_shap_vals):
                val = clean_shap_vals[i]
                if abs(val) > 1e-4:
                    weight = (val / total_logit_change) * abs(total_prob_movement)
                    impact_list.append({
                        "Factor": name,
                        "Points": round(float(weight), 1)
                    })

        df_impact = pd.DataFrame(impact_list)

        if df_impact.empty or "Points" not in df_impact.columns:
            df_impact = pd.DataFrame({"Factor": ["No Significant Factors"], "Points": [0.0]})
        else:
            df_impact = df_impact.sort_values(by="Points", key=lambda col: np.abs(col), ascending=False).head(8)
        # =========================================================
        # 7. OTHER FACTORS
        # =========================================================
        top_8_sum = df_impact["Points"].sum()
        other_factors_sum = round(total_prob_movement - top_8_sum, 1)

        # =========================================================
        # 8. NARRATIVE
        # =========================================================
        top_f = df_impact.iloc[0]

        st.info(
            f"""
            **The Risk Journey:** Starting from a baseline of
            **{base_prob_pct}%**,
            the model adjusts risk based on your profile.

            **{top_f['Factor']}**
            had the biggest impact,
            moving the score by
            **{abs(top_f['Points']):.1f} points**.
            """
        )

        # =========================================================
        # 9. WATERFALL CHART
        # =========================================================
        st.markdown("### 📊 How the Risk Accumulates")

        x_labels = (
            ["Baseline"]
            + df_impact["Factor"].tolist()
            + ["Other Factors", "Final Prediction"]
        )

        y_values = (
            [base_prob_pct]
            + df_impact["Points"].tolist()
            + [other_factors_sum, 0]
        )

        measures = (
            ["absolute"]
            + (["relative"] * (len(df_impact) + 1))
            + ["total"]
        )

        fig = go.Figure(go.Waterfall(
            name="Risk Journey",
            orientation="v",
            measure=measures,
            x=x_labels,
            textposition="outside",
            text=[
                f"{v}%"
                if m == "absolute"
                else f"{v:+.1f}%"
                for v, m in zip(y_values, measures)
            ],
            y=y_values,
            connector={"line": {"color": "rgb(63,63,63)"}},
            decreasing={"marker": {"color": "#00C853"}},
            increasing={"marker": {"color": "#FF4B4B"}},
            totals={"marker": {"color": "#00D4FF"}}, # Electric Cyan
            textfont=dict(color="white", size=12)
        ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            margin=dict(l=10, r=10, t=20, b=10),
            height=500,
            xaxis=dict(
                tickangle=-30,
                tickfont=dict(color="white"),
                color="white"
            ),
            yaxis=dict(
                title=dict(
                    text="Probability (%)",
                    font=dict(color="white")
                ),
                tickfont=dict(color="white"),
                gridcolor="#30363D",
                color="white"
            )
        )

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # =========================================================
        # 10. BREAKDOWN TABLE
        # =========================================================
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📋 Step-by-Step Breakdown")

        table_rows = [
            {
                "Factor": "🌎 General Baseline",
                "Impact": f"{base_prob_pct}% starting point"
            }
        ]

        for _, row in df_impact.iterrows():
            icon = "🔴" if row['Points'] > 0 else "🟢"
            direction = "Increase" if row['Points'] > 0 else "Decrease"

            table_rows.append({
                "Factor": row['Factor'],
                "Impact": f"{icon} {row['Points']:+.1f}% points {direction}"
            })

        other_icon = "🔴" if other_factors_sum > 0 else "🟢"
        other_dir = "Increase" if other_factors_sum > 0 else "Decrease"

        table_rows.append({
            "Factor": "Miscellaneous Other Factors",
            "Impact": f"{other_icon} {other_factors_sum:+.1f}% points {other_dir}"
        })

        table_rows.append({
            "Factor": "**Final Calculated Risk**",
            "Impact": f"**{prob:.2%}**"
        })

        st.table(pd.DataFrame(table_rows))

    except Exception as e:
        import traceback
        st.error(f"Display Error: {str(e)}")
        st.code(traceback.format_exc())

    # =========================================================
    # BACK BUTTON
    # =========================================================
    if st.button("⬅️", type="secondary"):
        st.session_state.current_step = 3
        st.rerun()