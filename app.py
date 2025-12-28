"""
app.py
======
Creep Strain Prediction and Monitoring System
Streamlit Web Application

Author: Senior Python + Streamlit Engineer
Purpose: UI layer connecting to existing ML backend
"""

import streamlit as st
import pandas as pd
from datetime import datetime

# Import backend modules (DO NOT MODIFY THESE)
from utils.predictor import CreepPredictor
from utils.alerts import (
    threshold_alert_check,
    trend_alert_check,
    fuse_alerts,
    get_operator_guidance,
    get_alert_color
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Creep Strain Monitoring",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def initialize_session_state():
    """
    Initialize all session state variables to prevent KeyErrors.
    Called once at app startup.
    """
    # Current page tracking
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'
    
    # Prediction history storage
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    # Strain history for trend analysis (keep last 5)
    if 'strain_history' not in st.session_state:
        st.session_state.strain_history = []
    
    # Latest prediction results
    if 'latest_duration' not in st.session_state:
        st.session_state.latest_duration = None
    if 'latest_voltage' not in st.session_state:
        st.session_state.latest_voltage = None
    if 'latest_strain' not in st.session_state:
        st.session_state.latest_strain = None
    if 'latest_alert' not in st.session_state:
        st.session_state.latest_alert = None

# Initialize session state
initialize_session_state()

# =============================================================================
# LOAD PREDICTOR (CACHED)
# =============================================================================

@st.cache_resource
def load_predictor():
    """
    Load the CreepPredictor once and cache it.
    This prevents reloading the model on every rerun.
    """
    try:
        predictor = CreepPredictor(
            model_path='xgb_model.pkl',
            scaler_path='scaler.pkl',
            config_path='config.pkl'
        )
        return predictor
    except Exception as e:
        st.error(f"Failed to load predictor: {str(e)}")
        st.stop()

# Load predictor
predictor = load_predictor()
config = predictor.get_config()

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================

def apply_custom_css():
    """
    Apply custom CSS styling for enhanced UI appearance.
    """
    st.markdown("""
    <style>
    /* Header background image with overlay */
    .header-container {
        position: relative;
        background-image: url('assets/header_bg.jpg');
        background-size: cover;
        background-position: center;
        padding: 80px 20px;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    
    .header-overlay {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: rgba(0, 0, 0, 0.5);
        border-radius: 10px;
    }
    
    .header-title {
        position: relative;
        text-align: center;
        color: white;
        font-size: 3em;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
        z-index: 1;
    }
    
    /* Navigation buttons */
    .nav-button-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 30px;
        flex-wrap: wrap;
    }
    
    /* Faded images */
    .faded-image {
        opacity: 0.7;
        border-radius: 10px;
        transition: opacity 0.3s;
    }
    
    .faded-image:hover {
        opacity: 1.0;
    }
    
    /* Alert boxes */
    .alert-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        font-size: 1.1em;
    }
    
    /* Metric styling */
.stMetric {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 8px;
}

/* Metric text color fix */
.stMetric label {
    color: #000000 !important;
}

.stMetric div {
    color: #000000 !important;
}

    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# =============================================================================
# HOME PAGE
# =============================================================================

def show_home_page():
    """
    Display the home page with header and navigation buttons.
    """
    # Header section with background image
    st.markdown("""
    <div class="header-container">
        <div class="header-overlay"></div>
        <h1 class="header-title">Creep Strain Prediction and Monitoring System</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Welcome message
    st.markdown("### Welcome to the Creep Strain Monitoring Platform")
    st.write("""
    This system uses advanced machine learning to predict creep strain in P-91 steel 
    under constant pressure and high-temperature conditions. Monitor material behavior, receive real-time alerts, 
    and make informed maintenance decisions.
    """)
    
    # Navigation buttons
    st.markdown("### Navigate to:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 Predict Here", use_container_width=True, type="primary"):
            st.session_state.current_page = 'predict'
            st.rerun()
    
    with col2:
        if st.button("📊 Prediction History", use_container_width=True):
            st.session_state.current_page = 'history'
            st.rerun()
    
    with col3:
        if st.button("ℹ️ About the Experiment", use_container_width=True):
            st.session_state.current_page = 'about'
            st.rerun()
    
    # System status
    st.markdown("---")
    st.markdown("### System Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Warning Threshold",
            value=f"{config['warning_threshold']:.2f}%"
        )
    
    with col2:
        st.metric(
            label="Model RMSE",
            value=f"{config['rmse']:.4f}%"
        )
    
    with col3:
        st.metric(
            label="Predictions Made",
            value=len(st.session_state.prediction_history)
        )

# =============================================================================
# PREDICT HERE PAGE
# =============================================================================

def show_predict_page():
    """
    Display the prediction page with input fields and results.
    """
    st.title("🔍 Creep Strain Prediction")
    
    # Back to home button
    if st.button("← Back to Home"):
        st.session_state.current_page = 'home'
        st.rerun()
    
    st.markdown("---")
    
    # Input section
    st.subheader("Input Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        duration = st.number_input(
            "Exposure Duration (Hours)",
            min_value=0.0,
            max_value=10000.0,
            value=100.0,
            step=10.0,
            help="Total time the material has been exposed to high temperature"
        )
    
    with col2:
        voltage = st.number_input(
            "Output Voltage (mV)",
            min_value=0.0,
            max_value=1000.0,
            value=250.0,
            step=10.0,
            help="Sensor output voltage measurement"
        )
    
    # Predict button
    if st.button("🚀 Predict Strain", type="primary", use_container_width=True):
        
        # Step 1: Validate inputs
        is_valid, error_msg = predictor.validate_input(duration, voltage)
        
        if not is_valid:
            st.error(f"❌ Input Validation Error: {error_msg}")
        else:
            with st.spinner("Calculating strain prediction..."):
                try:
                    # Step 2: Make prediction
                    strain = predictor.predict(duration, voltage)
                    
                    # Step 3: Update strain history (keep last 5 for trend analysis)
                    st.session_state.strain_history.append(strain)
                    if len(st.session_state.strain_history) > 5:
                        st.session_state.strain_history.pop(0)
                    
                    # Step 4: Compute alerts
                    threshold_status = threshold_alert_check(
                        strain, 
                        config['warning_threshold']
                    )
                    
                    trend_status = trend_alert_check(
                        st.session_state.strain_history
                    )
                    
                    final_alert = fuse_alerts(threshold_status, trend_status)
                    
                    # Step 5: Store in session state
                    st.session_state.latest_duration = duration
                    st.session_state.latest_voltage = voltage
                    st.session_state.latest_strain = strain
                    st.session_state.latest_alert = final_alert
                    
                    # Step 6: Add to prediction history
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.prediction_history.append({
                        'timestamp': timestamp,
                        'duration': duration,
                        'voltage': voltage,
                        'strain': strain,
                        'threshold_status': threshold_status,
                        'trend_status': trend_status,
                        'final_alert': final_alert
                    })
                    
                    # Success message
                    st.success("✅ Prediction completed successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Prediction Error: {str(e)}")
    
    # Display results section
    if st.session_state.latest_strain is not None:
        st.markdown("---")
        st.subheader("Prediction Results")
        
        # Display strain prediction
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Input Duration",
                value=f"{st.session_state.latest_duration:.2f} hrs"
            )
        
        with col2:
            st.metric(
                label="Input Voltage",
                value=f"{st.session_state.latest_voltage:.2f} mV"
            )
        
        with col3:
            st.metric(
                label="Predicted Strain",
                value=f"{st.session_state.latest_strain:.4f} %"
            )
        
        # Display alert status
        st.markdown("### Alert Status")
        
        final_alert = st.session_state.latest_alert
        alert_color = get_alert_color(final_alert)
        guidance = get_operator_guidance(final_alert)
        
        # Display alert with appropriate color
        if alert_color == "red":
            st.error(f"🚨 **{final_alert}**")
            st.error(guidance)
        elif alert_color == "orange":
            st.warning(f"⚠️ **{final_alert}**")
            st.warning(guidance)
        elif alert_color == "blue":
            st.info(f"ℹ️ **{final_alert}**")
            st.info(guidance)
        else:
            st.success(f"✅ **{final_alert}**")
            st.success(guidance)

# =============================================================================
# PREDICTION HISTORY PAGE
# =============================================================================

def show_history_page():
    """
    Display prediction history with table and charts.
    """
    st.title("📊 Prediction History")
    
    # Back to home button
    if st.button("← Back to Home"):
        st.session_state.current_page = 'home'
        st.rerun()
    
    st.markdown("---")
    
    # Check if history exists
    if len(st.session_state.prediction_history) == 0:
        st.info("No predictions yet. Go to 'Predict Here' to make your first prediction.")
        return
    
    # Convert history to DataFrame
    df = pd.DataFrame(st.session_state.prediction_history)
    
    # Display summary statistics
    st.subheader("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", len(df))
    
    with col2:
        st.metric("Avg Strain", f"{df['strain'].mean():.4f}%")
    
    with col3:
        st.metric("Max Strain", f"{df['strain'].max():.4f}%")
    
    with col4:
        critical_count = len(df[df['final_alert'] == 'CRITICAL ALERT'])
        st.metric("Critical Alerts", critical_count)
    
    # Display history table
    st.subheader("Prediction Records")
    
    # Format DataFrame for display
    display_df = df[['timestamp', 'duration', 'voltage', 'strain', 'final_alert']].copy()
    display_df.columns = ['Timestamp', 'Duration (hrs)', 'Voltage (mV)', 'Strain (%)', 'Alert Status']
    display_df['Strain (%)'] = display_df['Strain (%)'].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Charts section
    st.markdown("---")
    st.subheader("Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Exposure Duration vs % Strain")
        chart_data = pd.DataFrame({
            'Exposure Duration (hrs)': df['duration'],
            'Strain (%)': df['strain']
        })
        st.line_chart(chart_data.set_index('Exposure Duration (hrs)'))
    
    with col2:
        st.markdown("#### Output Voltage vs % Strain")
        chart_data = pd.DataFrame({
            'Output Voltage (mV)': df['voltage'],
            'Strain (%)': df['strain']
        })
        st.line_chart(chart_data.set_index('Output Voltage (mV)'))
    
    # Alert distribution
    st.markdown("---")
    st.subheader("Alert Distribution")
    
    alert_counts = df['final_alert'].value_counts()
    st.bar_chart(alert_counts)
    
    # Download option
    st.markdown("---")
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download History as CSV",
        data=csv,
        file_name=f"creep_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# =============================================================================
# ABOUT THE EXPERIMENT PAGE
# =============================================================================

def show_about_page():
    """
    Display information about the creep experiment.
    """
    st.title("ℹ️ About the Experiment")
    
    # Back to home button
    if st.button("← Back to Home"):
        st.session_state.current_page = 'home'
        st.rerun()
    
    st.markdown("---")
    
    # Purpose section
    st.subheader("Purpose")
    st.markdown("""
    This creep strain prediction and monitoring system can be applied in industries where materials operate under sustained high temperatures and mechanical stress. Key applications include condition monitoring of turbine blades, boiler components, and pressure vessels in thermal and nuclear power plants. The system also supports predictive maintenance in petrochemical and process industries by enabling early identification of creep-related degradation. Additionally, it can be used in materials research laboratories to analyze long-term creep behavior, validate experimental data, and assist in the development of high-temperature alloys with improved service life.
    """)
    
    st.markdown("---")
    
    # Applications section
    st.subheader("Applications")
    st.markdown("""
    The creep strain prediction system has wide-ranging applications across industries where materials are subjected to prolonged high-temperature and stress conditions. By enabling early detection and continuous monitoring of creep behavior, this system supports safer operation and informed decision-making.

Condition monitoring of turbine blades, boiler components, and pressure vessels in thermal and nuclear power plants

Predictive maintenance planning by detecting early-stage creep degradation under high-temperature operation

Structural health assessment in petrochemical, refinery, and process industries

Support for materials research laboratories in analyzing long-term creep behavior

Validation of experimental creep test data using machine learning predictions

Assistance in the design and development of high-temperature alloys with improved service life

Overall, the system enhances operational reliability, reduces the risk of catastrophic failures, and contributes to improved asset lifespan and safety in high-temperature engineering applications.
    """)
    
    st.markdown("---")
    
    # Images section
    st.subheader("Related Images")
    
    # Display images in columns with faded effect
    col1, col2, col3 = st.columns(3)
    
    with col1:
        try:
            st.markdown('<div class="faded-image">', unsafe_allow_html=True)
            st.image("assets/turbine blades.png", caption="Turbine Blades", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        except:
            st.warning("Image not found: turbine blades.png")
    
    with col2:
        try:
            st.markdown('<div class="faded-image">', unsafe_allow_html=True)
            st.image("assets/core-facilities.png", caption="Core Facilities", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        except:
            st.warning("Image not found: core-facilities.png")
    
    with col3:
        try:
            st.markdown('<div class="faded-image">', unsafe_allow_html=True)
            st.image("assets/6-CREEP.jpg", caption="Creep Testing", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        except:
            st.warning("Image not found: 6- CREEP.jpg")
    
    # Technical specifications
    st.markdown("---")
    st.subheader("Technical Specifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Model Details:**
        - Algorithm: XGBoost Regression
        - Input Features: Exposure Duration, Output Voltage
        - Output: Creep Strain Percentage
        - Training Data: P-91 Steel Creep Experiments
        """)
    
    with col2:
        st.markdown(f"""
        **Performance Metrics:**
        - Warning Threshold: {config['warning_threshold']:.2f}%
        - Model RMSE: {config['rmse']:.4f}%
        - Scaling Method: RobustScaler
        - Alert Types: Threshold + Trend Analysis
        """)

# =============================================================================
# MAIN APPLICATION ROUTER
# =============================================================================

def main():
    """
    Main application function that routes to appropriate page.
    """
        
    # Route to appropriate page
    if st.session_state.current_page == 'home':
        show_home_page()
    elif st.session_state.current_page == 'predict':
        show_predict_page()
    elif st.session_state.current_page == 'history':
        show_history_page()
    elif st.session_state.current_page == 'about':
        show_about_page()

# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()