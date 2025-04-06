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
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_url}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        font-family: 'Segoe UI', sans-serif;
    }}

    /* Apply translucent card style */
    .block-container {{
        background-color: rgba(255, 255, 255, 0.75);
        padding: 2rem 1rem;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
        margin: 2rem;
    }}

    h1 {{
        color: #0d3b66;
        text-shadow: 1px 1px 2px white;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
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

