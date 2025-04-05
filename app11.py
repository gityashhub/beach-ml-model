import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# App Title
st.title("🌊 Coastal Tourism Suitability Predictor")
st.markdown("Upload your dataset to train a model and predict tourism safety with confidence level.")

# Upload Dataset
uploaded_file = st.file_uploader("📂 Upload Coastal Tourism Dataset (.csv)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Dataset")
    st.dataframe(df.head())

    # Dataset Overview
    st.subheader("📋 Data Summary")
    st.write("Shape:", df.shape)
    st.write("Missing Values:\n", df.isnull().sum())

    # Target Visualization
    st.subheader("📈 Suitability Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Suitability", data=df, ax=ax)
    st.pyplot(fig)

    # Preprocessing
    st.subheader("⚙️ Preprocessing")
    df = pd.get_dummies(df, columns=["Beach", "Weather_Condition"])
    label_encoder = LabelEncoder()
    df["Suitability"] = label_encoder.fit_transform(df["Suitability"])

    X = df.drop(columns=["Suitability"])
    y = df["Suitability"]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    st.success(f"✅ Model trained successfully with accuracy: **{acc:.2f}**")

    # User Input for Prediction
    st.subheader("🧪 Predict Suitability")
    st.markdown("Fill in the feature values below:")

    user_input = {}
    for col in X.columns:
        if set(df[col].unique()) == {0, 1}:
            user_input[col] = st.selectbox(col, [0, 1])
        else:
            user_input[col] = st.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))

    input_df = pd.DataFrame([user_input])

    if st.button("Predict"):
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        confidence = np.max(probabilities) * 100

        st.success(f"🎯 Predicted Suitability: **{predicted_label}**")
        st.info(f"📊 Confidence Level: **{confidence:.2f}%**")

        # Show all class probabilities (optional)
        st.subheader("📍 Class Probabilities")
        prob_df = pd.DataFrame({
            "Class": label_encoder.inverse_transform(np.arange(len(probabilities))),
            "Probability (%)": (probabilities * 100).round(2)
        })
        st.dataframe(prob_df)

else:
    st.warning("👈 Please upload a CSV file to begin.")


# import streamlit as st
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler

# # === Load model ===
# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model("beach_safety_model.h5")

# model = load_model()

# # === Categories from dataset (adjust if needed) ===
# BEACHES = ['Bondi', 'Copacabana', 'Waikiki', 'Santa Monica']
# WEATHER_CONDITIONS = ['Sunny', 'Cloudy', 'Rainy', 'Stormy']

# # === The exact 18 features used during model training ===
# expected_columns = [
#     'Temperature', 'Wind_Speed', 'Wave_Height', 'UV_Index', 'Tide_Level',
#     'Beach_Bondi', 'Beach_Copacabana', 'Beach_Waikiki', 'Beach_Santa Monica',
#     'Weather_Condition_Sunny', 'Weather_Condition_Cloudy',
#     'Weather_Condition_Rainy', 'Weather_Condition_Stormy',
#     'Extra_Feature_1', 'Extra_Feature_2', 'Extra_Feature_3',
#     'Extra_Feature_4', 'Extra_Feature_5'  # 👈 Replace with your real column names
# ]

# # === Streamlit UI ===
# st.title("🏖️ Beach Safety Prediction App")
# st.write("Enter the current environmental conditions to predict safety status:")

# # User inputs
# temperature = st.slider("Temperature (°C)", 0.0, 45.0, 25.0)
# wind_speed = st.slider("Wind Speed (km/h)", 0.0, 100.0, 15.0)
# wave_height = st.slider("Wave Height (m)", 0.0, 5.0, 1.0)
# uv_index = st.slider("UV Index", 0.0, 11.0, 6.0)
# tide_level = st.slider("Tide Level (m)", 0.0, 3.0, 1.2)
# beach = st.selectbox("Select Beach", BEACHES)
# weather = st.selectbox("Weather Condition", WEATHER_CONDITIONS)

# # === Prediction ===
# if st.button("Predict"):
#     # Create input dataframe
#     input_data = pd.DataFrame([{
#         'Temperature': temperature,
#         'Wind_Speed': wind_speed,
#         'Wave_Height': wave_height,
#         'UV_Index': uv_index,
#         'Tide_Level': tide_level,
#         'Beach': beach,
#         'Weather_Condition': weather
#     }])

#     # One-hot encoding
#     input_data = pd.get_dummies(input_data, columns=['Beach', 'Weather_Condition'])

#     # Add missing columns with 0
#     for col in expected_columns:
#         if col not in input_data.columns:
#             input_data[col] = 0

#     # Reorder columns to match model
#     input_data = input_data[expected_columns]

#     # Debug shape check
#     st.caption(f"🧪 Input shape: {input_data.shape} (Model expects 18)")

#     # Predict
#     prediction = model.predict(input_data)[0][0]
#     if prediction > 0.5:
#         st.success(f"✅ SAFE for beach activities! (Confidence: {prediction:.2f})")
#     else:
#         st.error(f"⚠️ NOT SAFE today. (Confidence: {prediction:.2f})")
