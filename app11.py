import streamlit as st
import joblib
import numpy as np
import requests

# Load the model
model = joblib.load('beach_safety_model (1).pkl')

# Wind direction mapping
wind_dir_map = {'N': 0, 'NE': 1, 'E': 2, 'SE': 3, 'S': 4, 'SW': 5, 'W': 6, 'NW': 7}
def deg_to_compass(deg):
    directions = list(wind_dir_map.keys())
    idx = int((deg + 22.5) / 45.0) % 8
    return directions[idx]

# Fetch weather data
def fetch_weather_data(api_key, lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise ValueError(f"API returned status code {response.status_code}")
    
    data = response.json()
    try:
        temperature = data['main']['temp']
        humidity = data['main']['humidity']
        wind_speed = data['wind']['speed'] * 3.6  # m/s to km/h
        wind_deg = data['wind']['deg']
        wind_dir = deg_to_compass(wind_deg)
        visibility = data.get('visibility', 8000) / 1000
        rainfall = data.get('rain', {}).get('1h', 0)

        # Placeholder marine data
        wave_height = 1.0
        tide_level = 1.0
        uv_index = 5

        return {
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "wind_direction": wind_dir,
            "wave_height": wave_height,
            "tide_level": tide_level,
            "visibility": visibility,
            "rainfall": rainfall,
            "uv_index": uv_index
        }
    except KeyError as e:
        raise ValueError(f"Missing key in API response: {e}")

# Prediction function
def predict_beach_safety(data):
    wind_encoded = wind_dir_map.get(data['wind_direction'], 0)
    features = np.array([[data['temperature'], data['humidity'], data['wind_speed'],
                          wind_encoded, data['wave_height'], data['tide_level'],
                          data['visibility'], data['rainfall'], data['uv_index']]])
    return model.predict(features)[0]

# UI
st.set_page_config(page_title="Beach Safety Predictor", page_icon="🏖️")

st.title("🏖️ Beach Safety Predictor")
background_url = "https://images.unsplash.com/photo-1507525428034-b723cf961d3e"
st.markdown("""
<style>
/* App background */
.stApp {
    background-image: url('https://images.unsplash.com/photo-1507525428034-b723cf961d3e');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    font-family: 'Segoe UI', sans-serif;
    font-weight: bold;
    color: #003b49;
}

/* Glass effect container */
.main-container {
    background: rgba(255, 255, 255, 0.88);
    border-radius: 1.5rem;
    padding: 2.5rem;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.25);
    backdrop-filter: blur(10px);
    max-width: 780px;
    margin: 2rem auto;
}

/* Title */
h1 {
    text-align: center;
    color: #004e64;
    font-size: 3rem;
    font-weight: bold;
    margin-bottom: 1rem;
}

/* Subtitles */
h2, h3, h4 {
    font-size: 1.5rem !important;
    font-weight: bold !important;
    color: #006c80 !important;
}

/* Caption style */
.stMarkdown p {
    font-size: 1.1rem !important;
    font-weight: bold !important;
    color: #003b49 !important;
    margin-bottom: 0.5rem;
}

/* Label common style */
label, .stSelectbox label, .stNumberInput label, .stTextInput label {
    font-size: 1.1rem !important;
    font-weight: bold !important;
    color: #003b49 !important;
    margin-bottom: 0.3rem;
}

/* Inputs styling */
input, .stTextInput input, .stNumberInput input, .stPassword input {
    font-size: 1.05rem !important;
    font-weight: bold !important;
    background-color: rgba(255, 255, 255, 0.9);
    border-radius: 0.5rem;
    padding: 0.5rem;
    color: #003b49 !important;
}

# /* Select box styling */
# .stSelectbox div {
#     font-size: 1.05rem !important;
#     font-weight: bold !important;
#     color: #003b49 !important;
#     background-color: rgba(255, 255, 255, 0.85);
#     border-radius: 0.5rem !important;
#     padding: 0.4rem;
# }

/* Radio buttons label */
.stRadio label {
    font-size: 1.1rem !important;
    font-weight: bold !important;
    color: #004e64;
}

/* Button style */
.stButton > button {
    background-color: #0096a3;
    color: white;
    font-weight: bold;
    font-size: 1.1rem;
    padding: 0.6rem 1.2rem;
    border: none;
    border-radius: 0.8rem;
    transition: all 0.3s ease;
    margin-top: 1rem;
}

.stButton > button:hover {
    background-color: #007d8a;
    transform: scale(1.02);
}

/* Result success box */
.stAlert-success {
    background-color: #d4f4dd !important;
    color: #006c3f !important;
    border-left: 5px solid #30c769 !important;
}

/* Result error box */
.stAlert-error {
    background-color: #ffe0e0 !important;
    color: #a80000 !important;
    border-left: 5px solid #ff4d4d !important;
}

/* Align everything centrally */
.block-container {
    display: flex;
    flex-direction: column;
    align-items: center;
}
</style>
""", unsafe_allow_html=True)



st.caption("Predict if the beach is safe based on manual data or real-time weather info.")

mode = st.radio("Choose mode:", ["📝 Manual Input", "🌐 Real-Time Weather"])

if mode == "📝 Manual Input":
    st.subheader("🔧 Enter Beach Conditions Manually")
    temperature = st.number_input("🌡️ Temperature (°C)", 0.0, 50.0, 28.0)
    humidity = st.number_input("💧 Humidity (%)", 0.0, 100.0, 70.0)
    wind_speed = st.number_input("💨 Wind Speed (km/h)", 0.0, 100.0, 15.0)
    wind_direction = st.selectbox("🧭 Wind Direction", list(wind_dir_map.keys()))
    wave_height = st.number_input("🌊 Wave Height (m)", 0.0, 10.0, 1.2)
    tide_level = st.number_input("🌊 Tide Level (m)", 0.0, 5.0, 0.8)
    visibility = st.number_input("👁️ Visibility (km)", 0.0, 20.0, 8.0)
    rainfall = st.number_input("🌧️ Rainfall (mm)", 0.0, 100.0, 0.0)
    uv_index = st.number_input("☀️ UV Index", 0.0, 15.0, 7.0)

    if st.button("🔍 Predict"):
        data = {
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "wind_direction": wind_direction,
            "wave_height": wave_height,
            "tide_level": tide_level,
            "visibility": visibility,
            "rainfall": rainfall,
            "uv_index": uv_index
        }
        result = predict_beach_safety(data)
        st.subheader("🔎 Prediction Result")
        st.success("✅ Beach is SAFE to visit!" if result == 1 else "⚠️ Beach is NOT safe to visit!")

elif mode == "🌐 Real-Time Weather":
    st.subheader("🌦️ Use Live Weather Data")
    lat = st.number_input("📍 Latitude", value=19.0760)
    lon = st.number_input("📍 Longitude", value=72.8777)
    api_key = st.text_input("🔑 OpenWeatherMap API Key", type="password")

    if st.button("⚡ Fetch & Predict"):
        if not api_key:
            st.warning("Please provide a valid OpenWeatherMap API key.")
        else:
            try:
                data = fetch_weather_data(api_key, lat, lon)
                st.json(data)
                result = predict_beach_safety(data)
                st.subheader("🔎 Prediction Result")
                st.success("✅ Beach is SAFE to visit!" if result == 1 else "⚠️ Beach is NOT safe to visit!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
