import streamlit as st
import pandas as pd


st.set_page_config(
    page_title="Pydaling Paris",
    layout="centered", 
    initial_sidebar_state="expanded"
)

# Import page modules
from introduction import show_introduction
from data_exploration import data_exploration_page
from data_visualization import data_visualization_page
from data_modeling import render_data_modeling_page
from prediction import prediction_page
from conclusion import render_conclusion_page
from utils import format_dataframe 


# Load datasets from local
bikes_paris_path = "C:/Users/julia/OneDrive/Documents/04_DataScientest/01_Data Analyst/Bikes_Paris/01_Data/01_External/comptage-velo-donnees-compteurs.csv"
bikes_paris_clean_path = "C:/Users/julia/OneDrive/Documents/04_DataScientest/01_Data Analyst/Bikes_Paris/01_Data/02_Cleaned/bike_df_cleaned.csv"


if 'bikes_paris' not in st.session_state:
    try:
        st.session_state.bikes_paris = pd.read_csv(bikes_paris_path, sep=';')
    except FileNotFoundError:
        st.session_state.bikes_paris = None
        st.sidebar.error(f"Raw data file not found. Please check the path: {bikes_paris_path}")
    except Exception as e:
        st.session_state.bikes_paris = None
        st.sidebar.error(f"Error reading raw data file: {e}")

if 'bikes_paris_clean' not in st.session_state:
    try:
        st.session_state.bikes_paris_clean = pd.read_csv(bikes_paris_clean_path, sep=',')
    except FileNotFoundError:
        st.session_state.bikes_paris_clean = None
        st.sidebar.error(f"Cleaned data file not found. Please check the path: {bikes_paris_clean_path}")
    except Exception as e:
        st.session_state.bikes_paris_clean = None
        st.sidebar.error(f"Error reading cleaned data file: {e}")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Introduction", 
    "Data Exploration and Preprocessing", 
    "Data Visualization", 
    "Data Modelling", 
    "Prediction", 
    "Conclusion"
])


if page == "Introduction":
    show_introduction()
elif page == "Data Exploration and Preprocessing":
    data_exploration_page()
elif page == "Data Visualization":
    data_visualization_page()
elif page == "Data Modelling":
    render_data_modeling_page(bikes_paris_clean_path)
elif page == "Prediction":
    prediction_page()
elif page == "Conclusion":
    render_conclusion_page()