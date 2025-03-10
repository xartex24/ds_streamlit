import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

@st.cache_data
def load_and_clean_data(parquet_path: str):
    # Load data from parquet
    df_local = pd.read_parquet(parquet_path)
    
    # Rename dictionary
    rename_dict = {
        'Identifiant du compteur': 'Cntr_ID',
        'Nom du compteur': 'Cntr_Name',
        'Identifiant du site de comptage': 'Cntg_Site_ID',
        'Nom du site de comptage': 'Cntg_Site_Name',
        'Comptage horaire': 'Hrly_Cnt',
        'Date et heure de comptage': 'Cntg_Date_Time',
        "Date d'installation du site de comptage": 'Cntg_Site_Inst_Date',
        'Lien vers photo du site de comptage': 'Link_Cntg_Site_Photo',
        'Coordonnées géographiques': 'Geo_Coord',
        'Identifiant technique compteur': 'Tech_Cntr_ID',
        'ID Photos': 'Photo_ID',
        'test_lien_vers_photos_du_site_de_comptage_': 'Test_Link_Cntg_Site_Photo',
        'id_photo_1': 'Photo_ID_1',
        'url_sites': 'Site_URLs',
        'type_dimage': 'Img_type',
        'mois_annee_comptage': 'Cntg_Month_Year'
    }
    df_local.rename(columns=rename_dict, inplace=True)

    # Convert 'Cntg_Date_Time' to datetime (UTC)
    df_local['Cntg_Date_Time'] = pd.to_datetime(df_local['Cntg_Date_Time'], errors='coerce', utc=True)
    df_local.dropna(subset=['Cntg_Date_Time'], inplace=True)

    # Extract hour
    df_local['hour'] = df_local['Cntg_Date_Time'].apply(lambda x: x.hour)

    # Split Geo_Coord
    df_local[['latitude', 'longitude']] = df_local['Geo_Coord'].str.split(',', expand=True)
    df_local['latitude'] = pd.to_numeric(df_local['latitude'], errors='coerce')
    df_local['longitude'] = pd.to_numeric(df_local['longitude'], errors='coerce')

    # Drop rows missing essential columns
    df_local.dropna(subset=['hour', 'latitude', 'longitude', 'Hrly_Cnt'], inplace=True)
    
    return df_local

@st.cache_resource
def load_model(model_path: str):
    with open(model_path, "rb") as f:
        return pickle.load(f)

# Load and clean data
df = load_and_clean_data("data/bikes_paris.parquet")

# Prepare features and target
X = df[['hour', 'latitude', 'longitude']]
y = df['Hrly_Cnt']

# Sidebar: Hyperparameter selection
st.sidebar.header("Random Forest Model Settings")
n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 10, 200, 50, 10)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 10, 1)
max_depth = st.sidebar.slider("Maximum Depth (max_depth)", 1, 50, 20, 1)
test_size = st.sidebar.slider("Test Set Size (%)", 10, 50, 20, 5)

st.sidebar.header("Select Model Option")
model_option = st.sidebar.radio(
    label="Choose what to do:",
    options=["Load Pre-trained Model", "Retrain Model"],
    index=0
)

def train_and_evaluate_rf(X, y, n_estimators, min_samples_split, max_depth, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
    model_local = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        max_depth=max_depth,
        random_state=42
    )
    model_local.fit(X_train, y_train)
    y_pred_local = model_local.predict(X_test)
    mae_local = mean_absolute_error(y_test, y_pred_local)
    mse_local = mean_squared_error(y_test, y_pred_local)
    rmse_local = np.sqrt(mse_local)
    r2_local = r2_score(y_test, y_pred_local)
    return model_local, mae_local, mse_local, rmse_local, r2_local, X_test, y_test, y_pred_local

if model_option == "Retrain Model":
    model, mae, mse, rmse, r2, X_test, y_test, y_pred = train_and_evaluate_rf(
        X, y, n_estimators, min_samples_split, max_depth, test_size
    )
    with open("models/rf_model.pkl", "wb") as f:
        pickle.dump(model, f)
    st.success("Model retrained and saved!")
else:
    model = load_model("models/rf_model.pkl")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
    y_pred = model.predict(X_test.to_numpy())
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    st.success("Pre-trained model loaded!")

st.subheader("Model Evaluation (with geographic data)")
st.write(f"Mean Absolute Error (MAE): {mae:.3f}")
st.write(f"Mean Squared Error (MSE): {mse:.3f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
st.write(f"R² Score: {r2:.3f}")
best_r2_val = 0.875
st.write(f"Best R² Score (Geographic Model): {best_r2_val:.3f}")

# Create an interactive 3D scatter plot using Plotly
fig = go.Figure(data=[go.Scatter3d(
    x=X_test["hour"],
    y=y_test,
    z=y_pred,
    mode='markers',
    marker=dict(
        size=10,
        color=y_test,
        colorscale='Viridis',
        opacity=0.8
    ),
    text=[f"Hour: {h}<br>Actual: {a}<br>Predicted: {p:.2f}"
          for h, a, p in zip(X_test["hour"], y_test, y_pred)],
    hoverinfo='text'
)])
fig.update_layout(
    scene=dict(
        xaxis_title='Hour',
        yaxis_title='Actual Traffic',
        zaxis_title='Predicted Traffic'
    ),
    margin=dict(l=0, r=0, b=0, t=0)
)
st.subheader("Interactive 3D Visualization")
st.plotly_chart(fig, use_container_width=True)

# Now show the full dataset sample below the 3D visualization
st.subheader("Full Dataset Sample")
st.dataframe(df.head())

# Real-time prediction widget in the sidebar
st.sidebar.subheader("Real-time Prediction")

unique_coords = df[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
unique_coords["coord_str"] = unique_coords.apply(lambda row: f"({row['latitude']:.4f}, {row['longitude']:.4f})", axis=1)
coord_options = unique_coords["coord_str"].tolist()

selected_coord_str = st.sidebar.selectbox("Select Location (latitude, longitude)", coord_options)
selected_coord_str = selected_coord_str.strip("()")
lat_str, lon_str = selected_coord_str.split(",")
input_latitude = float(lat_str)
input_longitude = float(lon_str)

input_hour = st.sidebar.number_input("Enter Hour (0-23)", min_value=0, max_value=23, value=12, step=1)

input_df = pd.DataFrame({
    "hour": [input_hour],
    "latitude": [input_latitude],
    "longitude": [input_longitude]
})

prediction_rt = model.predict(input_df.to_numpy())
st.sidebar.write(f"Predicted traffic: {prediction_rt[0]:.2f}")