import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
from PIL import Image

def load_data(bikes_paris_clean_path=None):
    """Load data from path or session state."""
    # If data is already in session state, use that
    if "df" in st.session_state:
        return st.session_state["df"]
    
    # If path is provided, try to load from that path
    if bikes_paris_clean_path is not None and os.path.exists(bikes_paris_clean_path):
        try:
            df = pd.read_csv(bikes_paris_clean_path)
            st.session_state["df"] = df
            return df
        except Exception as e:
            st.error(f"Error loading data from {bikes_paris_clean_path}: {e}")
    
    # Fallback to creating sample data
    st.warning("No data found. Creating sample data for demonstration.")
    # Create sample data similar to what would be in your cleaned dataframe
    data = {
        'hour': np.random.randint(0, 24, 1000),
        'longitude': np.random.uniform(2.2, 2.4, 1000),
        'latitude': np.random.uniform(48.8, 48.9, 1000),
        'Month': np.random.randint(1, 13, 1000),
        'Day': np.random.randint(1, 32, 1000),
        'Time_Of_Day': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], 1000),
        'Weekday_Number': np.random.randint(0, 7, 1000),
        'Hrly_Cnt': np.random.randint(10, 300, 1000)
    }
    df = pd.DataFrame(data)
    st.session_state["df"] = df
    return df

def render_data_modeling_page(bikes_paris_clean_path=None):
    """Render the data modeling page in the Streamlit app."""
    st.header("Data Modelling")
    st.write("In this section, we will go over multiple trained models and analyze their results")
    
    # Load data
    df = load_data(bikes_paris_clean_path)
    
    # Display a sample of the data if requested
    with st.expander("View sample data"):
        st.dataframe(df.head())
    
    # Select features and target
    selected_features = ['hour', 'longitude', 'latitude', 'Month', 'Day', 'Time_Of_Day', 'Weekday_Number']
    target_column = 'Hrly_Cnt'
    
    # Check if all features exist in the dataframe
    missing_features = [col for col in selected_features if col not in df.columns]
    if missing_features:
        st.error(f"Missing features in dataset: {missing_features}")
        return
    
    X = df[selected_features]
    y = df[target_column]
    
    # Handle Categorical Variables
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Handle Missing Values
    imputer = SimpleImputer(strategy="mean")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)
    
    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        ":chart_with_downwards_trend: Linear Regression", 
        ":deciduous_tree: Random Forest", 
        ":evergreen_tree: Decision Tree",
        ":large_orange_diamond: K-Means",
        ":large_blue_circle: DBSCAN"
    ])
    
    with tab1:  # LINEAR REGRESSION MODEL
        st.subheader("Linear Regression")
        
        with st.spinner("Training Linear Regression model..."):
            lr = LinearRegression()
            lr.fit(X_train_scaled, y_train)
            y_pred_lr = lr.predict(X_test_scaled)
            
            # Calculate metrics
            mae_lr = mean_absolute_error(y_test, y_pred_lr)
            rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
            r2_train_lr = lr.score(X_train_scaled, y_train)
            r2_test_lr = lr.score(X_test_scaled, y_test)
            
            # Performance Table
            table_lr = pd.DataFrame({
                " ": ["Incl. Geo-data", "Excl. Geo-data"],
                "MAE": [round(mae_lr, 2), 56.01],
                "RMSE": [round(rmse_lr, 2), 92.07],
                "Train R²-Score": [round(r2_train_lr, 3), 0.202],
                "Test R²-Score": [round(r2_test_lr, 3), 0.203]
            })
            
            st.markdown(table_lr.to_html(index=False), unsafe_allow_html=True)
            
            # Charts
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig1, ax1 = plt.subplots(figsize=(3, 2))
                sns.scatterplot(x=y_test, y=y_pred_lr, alpha=0.5, ax=ax1)
                ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                ax1.set_title("LR: Actual vs Predicted", fontsize=7)
                st.pyplot(fig1)
        
            with col2:
                fig2, ax2 = plt.subplots(figsize=(3, 2))
                sns.histplot(y_test - y_pred_lr, bins=20, kde=True, ax=ax2)
                ax2.set_title("LR: Residuals", fontsize=7)
                st.pyplot(fig2)
            
            with col3:
                fig3, ax3 = plt.subplots(figsize=(4, 3))
                feature_importance = pd.Series(np.abs(lr.coef_), index=X.columns)
                feature_importance = feature_importance.sort_values(ascending=False)
            
                # Create horizontal bar chart for better label visibility
                sns.barplot(x=feature_importance.values, y=feature_importance.index, orient='h', palette='Blues_d', ax=ax3)
                
                ax3.set_title("LR: Feature Importance", fontsize=10)
                ax3.set_xlabel("Importance", fontsize=8)
                ax3.tick_params(axis='y', labelsize=8)
                
                plt.tight_layout()
                st.pyplot(fig3)
            
            st.markdown("""
            **Key takeaways on this model:**
            - The model reveals a systematic underestimation of high bike counts.
            - The residuals histogram shows a right-skewed distribution, confirming that the model underestimates high values.
            - The Feature Importance chart shows a dominance of geographical features, specifically longitude, highlighting the model's over-reliance on spatial data.
            """)
    
    with tab2:  # RANDOM FOREST MODEL
        st.subheader("Random Forest")
        
        with st.spinner("Training Random Forest model..."):
            rf = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=20, min_samples_split=10, n_jobs=-1)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            
            # Calculate metrics
            mae_rf = mean_absolute_error(y_test, y_pred_rf)
            rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
            r2_train_rf = rf.score(X_train, y_train)
            r2_test_rf = rf.score(X_test, y_test)
            
            # Performance Table
            table_rf = pd.DataFrame({
                " ": ["Incl. Geo-data", "Excl. Geo-data"],
                "MAE": [round(mae_rf, 2), 20.81],
                "RMSE": [round(rmse_rf, 2), 44],
                "Train R²-Score": [round(r2_train_rf, 3), 0.824],
                "Test R²-Score": [round(r2_test_rf, 3), 0.818]
            })
            
            st.markdown(table_rf.to_html(index=False), unsafe_allow_html=True)
            
            # Charts
            col4, col5, col6 = st.columns(3)
            
            with col4:
                fig1, ax1 = plt.subplots(figsize=(3, 2))
                sns.scatterplot(x=y_test, y=y_pred_rf, alpha=0.5, ax=ax1)
                ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                ax1.set_title("RF: Actual vs Predicted", fontsize=7)
                st.pyplot(fig1)
    
            with col5:
                fig2, ax2 = plt.subplots(figsize=(3, 2))
                sns.histplot(y_test - y_pred_rf, bins=20, kde=True, ax=ax2)
                ax2.set_title("RF: Residuals", fontsize=7)
                st.pyplot(fig2)
            
            with col6:
                fig3, ax3 = plt.subplots(figsize=(4, 3))
                feature_importance = pd.Series(rf.feature_importances_, index=X.columns)
                feature_importance = feature_importance.sort_values(ascending=False)
                
                # Create horizontal bar chart for better label visibility
                sns.barplot(x=feature_importance.values, y=feature_importance.index, orient='h', palette='Blues_d', ax=ax3)
                
                ax3.set_title("RF: Feature Importance", fontsize=10)
                ax3.set_xlabel("Importance", fontsize=8)
                ax3.tick_params(axis='y', labelsize=8)
                
                plt.tight_layout()
                st.pyplot(fig3)
            
            st.markdown("""
            **Key takeaways on this model:**
            - The Actual vs. Predicted plot shows tight alignment along the ideal line, indicating high accuracy and minimal bias.
            - The Residuals Histogram demonstrates a symmetrical and narrow distribution, confirming consistent accuracy.
            - The Feature Importance Chart highlights the dominance of Hour, along with significant contributions from Longitude and Latitude, showing that spatial patterns complement temporal dependencies.
            """)
    
    with tab3:  # DECISION TREE MODEL
        st.subheader("Decision Tree")
        
        with st.spinner("Training Decision Tree model..."):
            dt = DecisionTreeRegressor(max_depth=20, random_state=42)
            dt.fit(X_train, y_train)
            y_pred_dt = dt.predict(X_test)
            
            # Calculate metrics
            mae_dt = mean_absolute_error(y_test, y_pred_dt)
            rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
            r2_train_dt = dt.score(X_train, y_train)
            r2_test_dt = dt.score(X_test, y_test)
            
            # Performance Table
            table_dt = pd.DataFrame({
                " ": ["Incl. Geo-data", "Excl. Geo-data"],
                "MAE": [round(mae_dt, 2), 31.38],
                "RMSE": [round(rmse_dt, 2), 59.64],
                "Train R²-Score": [round(r2_train_dt, 3), 0.764],
                "Test R²-Score": [round(r2_test_dt, 3), 0.665]
            })
            
            st.markdown(table_dt.to_html(index=False), unsafe_allow_html=True)
            
            # Charts
            col7, col8, col9 = st.columns(3)
            
            with col7:
                fig1, ax1 = plt.subplots(figsize=(3, 2))
                sns.scatterplot(x=y_test, y=y_pred_dt, alpha=0.5, ax=ax1)
                ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                ax1.set_title("DT: Actual vs Predicted", fontsize=7)
                st.pyplot(fig1)
    
            with col8:
                fig2, ax2 = plt.subplots(figsize=(3, 2))
                sns.histplot(y_test - y_pred_dt, bins=20, kde=True, ax=ax2)
                ax2.set_title("DT: Residuals", fontsize=7)
                st.pyplot(fig2)
            
            with col9:
                fig3, ax3 = plt.subplots(figsize=(4, 3))
                feature_importance = pd.Series(dt.feature_importances_, index=X.columns)
                feature_importance = feature_importance.sort_values(ascending=False)
                
                # Create horizontal bar chart for better label visibility
                sns.barplot(x=feature_importance.values, y=feature_importance.index, orient='h', palette='Blues_d', ax=ax3)
                
                ax3.set_title("DT: Feature Importance", fontsize=10)
                ax3.set_xlabel("Importance", fontsize=8)
                ax3.tick_params(axis='y', labelsize=8)
                
                plt.tight_layout()
                st.pyplot(fig3)
            
            st.markdown("""
            **Key takeaways on this model:**
            - The Actual vs. Predicted plot shows moderate alignment with the ideal line, indicating reasonable accuracy but with higher variance and bias in peak values.
            - The Residuals Histogram demonstrates a symmetrical but wider distribution, confirming moderate variance and inconsistent generalization.
            - The Feature Importance Chart highlights the dominance of Hour, along with significant contributions from Longitude and Latitude, showing that the model overfits to spatial and hourly patterns.
            - The model accurately captures hourly and spatial dependencies but struggles with non-linear interactions, leading to higher variance and overfitting.
            """)
    
    with tab4:  # K-Means Clustering
        st.subheader("K-Means")       
        # Try to display images if they exist
        try:
            img1_path = os.path.join("03_Exports", "Elbow_method.jpg")
            if os.path.exists(img1_path):
                img1 = Image.open(img1_path)
                img1_resized = img1.resize((1000, 800))  # Resize
                st.image(img1_resized, caption="Elbow Method Analysis")
                
            img2_path = os.path.join("03_Exports", "Kmeans_cluster.jpg")
            if os.path.exists(img2_path):
                img2 = Image.open(img2_path)
                img2_resized = img2.resize((1000, 800))  # Resize
                st.image(img2_resized, caption="K-Means Clusters")
        except Exception as e:
            st.warning(f"Could not display images: {e}")
        
        st.markdown("""
        **Key takeaways on this model:**
        - Diminishing Returns: Since adding more clusters does not significantly reduce within-cluster variance (inertia), the slight improvement does not justify the added complexity.
        - Overfitting Risk: Increasing the number of clusters can lead to overfitting, capturing noise or outliers rather than meaningful patterns.
        - Interpretability: By choosing 6 clusters, we maintain a more interpretable model, making it easier to analyze and communicate the results.
        - Elbow Method Analysis: Our Elbow Method analysis likely showed a clear bend at 6 clusters, indicating that most variance is captured with this number, and additional clusters provide diminishing returns.
        - Simplicity and Generalization: Using fewer clusters helps our model generalize better to new data by avoiding over-segmentation.
        """)
        
    with tab5:  # DBSCAN Clustering
        st.subheader("DBSCAN")
        
        st.markdown("""
        The DBSCAN clustering visualization would normally appear here. To display it, please:

        1. Ensure the image file "03_Exports/DBSCAN.jpg" exists in your project directory

        **Key takeaways on this model:**
        - Scalability Issue: DBSCAN's computational complexity increases significantly with large datasets, leading to long processing times.
        - It does not effectively differentiate the load levels of zones in our data, as it struggles with clusters of varying densities.
        - DBSCAN is highly sensitive to the choice of eps and min_samples, which is challenging to optimize for complex urban cycling patterns.
        - Given the high dimensionality of our data (e.g., time, location, and flow features), DBSCAN's performance degrades, affecting clustering quality.
        """)
        
        # Try to display image if it exists
        try:
            img3_path = os.path.join("03_Exports", "DBSCAN.jpg")
            if os.path.exists(img3_path):
                img3 = Image.open(img3_path)
                img3_resized = img3.resize((1000, 800))  # Resize
                st.image(img3_resized, caption="DBSCAN Clustering")
        except Exception as e:
            st.warning(f"Could not display images: {e}")

