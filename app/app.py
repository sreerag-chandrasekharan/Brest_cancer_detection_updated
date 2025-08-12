import streamlit as st
import pickle
import pandas as pd
import json
import plotly.graph_objects as go
import numpy as np

#-- load min max value for slidebar ----
with open("models/feature_stats.json", "r") as f:
    FEATURE_STATS = json.load(f)

# Top 5 features to show initially
TOP5_KEYS = [
    "perimeter_worst", "area_worst", "radius_worst", "radius_mean", "perimeter_mean"
]

# All 30 features and labels (same order as original)
SLIDER_LABELS = [
    ("Radius (mean)", "radius_mean"),
    ("Texture (mean)", "texture_mean"),
    ("Perimeter (mean)", "perimeter_mean"),
    ("Area (mean)", "area_mean"),
    ("Smoothness (mean)", "smoothness_mean"),
    ("Compactness (mean)", "compactness_mean"),
    ("Concavity (mean)", "concavity_mean"),
    ("Concave points (mean)", "concave points_mean"),
    ("Symmetry (mean)", "symmetry_mean"),
    ("Fractal dimension (mean)", "fractal_dimension_mean"),
    ("Radius (se)", "radius_se"),
    ("Texture (se)", "texture_se"),
    ("Perimeter (se)", "perimeter_se"),
    ("Area (se)", "area_se"),
    ("Smoothness (se)", "smoothness_se"),
    ("Compactness (se)", "compactness_se"),
    ("Concavity (se)", "concavity_se"),
    ("Concave points (se)", "concave points_se"),
    ("Symmetry (se)", "symmetry_se"),
    ("Fractal dimension (se)", "fractal_dimension_se"),
    ("Radius (worst)", "radius_worst"),
    ("Texture (worst)", "texture_worst"),
    ("Perimeter (worst)", "perimeter_worst"),
    ("Area (worst)", "area_worst"),
    ("Smoothness (worst)", "smoothness_worst"),
    ("Compactness (worst)", "compactness_worst"),
    ("Concavity (worst)", "concavity_worst"),
    ("Concave points (worst)", "concave points_worst"),
    ("Symmetry (worst)", "symmetry_worst"),
    ("Fractal dimension (worst)", "fractal_dimension_worst"),
]

# Use these to simulate the loaded data for scaling, as in your get_scaled_values_dict
def get_min_max_mean(key):
    return FEATURE_STATS[key]

# Sidebar with top 5 visible, rest in dropdown
def add_sidebar():
    st.sidebar.header("Measurements from Cytology lab")

    input_dict = {}

    # Show top 5 sliders directly
    for label, key in SLIDER_LABELS:
        if key in TOP5_KEYS:
            min_val, mean_val, max_val = get_min_max_mean(key)
            input_dict[key] = st.sidebar.slider(
                label,
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(mean_val),
                key=key,
            )

    # Dropdown for remaining sliders
    with st.sidebar.expander("Show more parameters"):
        for label, key in SLIDER_LABELS:
            if key not in TOP5_KEYS:
                min_val, mean_val, max_val = get_min_max_mean(key)
                input_dict[key] = st.slider(
                    label,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(mean_val),
                    key=key,
                )

    return input_dict

# scale values depending on hardcoded min and max

def get_scaled_values_dict(values_dict):
    scaled_dict = {}
    for key, value in values_dict.items():
        min_val, mean_val, max_val = get_min_max_mean(key)
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    return scaled_dict

# ----Import the scaler and model

model = pickle.load(open('models/model.pkl', "rb"))
scaler = pickle.load(open('models/scaler.pkl', "rb"))

#------- Radar chart ------
def add_radar_chart(input_data):
    input_data = get_scaled_values_dict(input_data)

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=[input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
                input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
                input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
                input_data['fractal_dimension_mean']],
            theta=['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                   'Symmetry', 'Fractal Dimension'],
            fill='toself',
            name='Mean'
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=[input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
                input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
                input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']],
            theta=['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                   'Symmetry', 'Fractal Dimension'],
            fill='toself',
            name='Standard Error'
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=[input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
                input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
                input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
                input_data['fractal_dimension_worst']],
            theta=['Radius', 'Texture', 'Perimeter', 'Area', 'Smoothness', 'Compactness', 'Concavity', 'Concave Points',
                   'Symmetry', 'Fractal Dimension'],
            fill='toself',
            name='Worst'
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        autosize=True
    )

    return fig

# ----Display predictions -----


FEATURE_ORDER = [key for _, key in SLIDER_LABELS]

def display_predictions(input_data):
    # build in fixed order
    row = [input_data[k] for k in FEATURE_ORDER]
    input_array = np.array(row).reshape(1, -1)

    input_data_scaled = scaler.transform(input_array)
    proba = model.predict_proba(input_data_scaled)[0]
    pred = model.predict(input_data_scaled)[0]

    st.subheader('Cell cluster prediction')
    st.write("The cell cluster is: ")
    if pred == 0:
        st.write("<span style='color:green; font-size:24px;'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span style='color:red;font-size:24px;'>Malignant</span>", unsafe_allow_html=True)

    st.write(f"Probability of being benign: {proba[0]:.2%}")
    st.write(f"Probability of being malignant: {proba[1]:.2%}")

    #st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")


# ----- main function to run the app -----
def main():
    st.set_page_config(page_title="Breast Cancer Diagnosis",
        page_icon=":microscope:", 
        layout="wide", 
        initial_sidebar_state="expanded")
    
    input_data = add_sidebar()

    # Set up the structure
    with st.container():
        st.title("Breast Cancer Detection App")
        st.write("This app uses a machine learning model to determine whether a breast mass is benign or malignant based on measurements from your cytology lab. Use the sliders in the sidebar to enter your measurements.")
        col1, col2 = st.columns([4,2])
        with col1:
            radar_chart = add_radar_chart(input_data)
            st.plotly_chart(radar_chart, use_container_width=True)
        with col2:
            display_predictions(input_data)



if __name__ == '__main__':
    main()