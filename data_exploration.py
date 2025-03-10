import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils import format_dataframe

def data_exploration_page():
    
    st.header("Data Exploration and Preprocessing")
    

    st.markdown("""
        ### Transforming Raw Cycling Data into Analytical Insights
        
        This section guides you through our data preprocessing journey - from raw bike counter 
        data to analysis-ready datasets. Discover how we clean, transform, and engineer features
        to enable meaningful analysis of Paris cycling patterns.
        """)

    
    # Tabs for different exploration aspects
    tab1, tab2, tab3 = st.tabs([
        "Raw Dataset", 
        "Preprocessing Steps", 
        "Cleaned Dataset"
    ])
    
    # Raw Dataset
    with tab1:
        raw_dataset_exploration()
    
    # Preprocessing Steps
    with tab2:
        preprocessing_steps()
    
    # Cleaned Dataset
    with tab3:
        cleaned_dataset_exploration()

def raw_dataset_exploration():
    if 'bikes_paris' in st.session_state and st.session_state.bikes_paris is not None:
        st.subheader("Raw Dataset Overview")
         
        total_rows = st.session_state.bikes_paris.shape[0]
        total_cols = st.session_state.bikes_paris.shape[1]
        missing_values = st.session_state.bikes_paris.isnull().sum().sum()
        missing_percentage = (missing_values / (total_rows * total_cols)) * 100
        
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{total_rows:,}", help="Number of records in the dataset")
        with col2:
            st.metric("Total Columns", total_cols, help="Number of features/variables")
        with col3:
            st.metric("Missing Values", f"{missing_values:,} ({missing_percentage:.2f}%)", 
                     help="Count and percentage of missing data points")
        
        
        st.subheader("Dataset Inspection")
        
        
        inspect_tab1, inspect_tab2, inspect_tab3, inspect_tab4 = st.tabs([
            "Sample Data", "Column Info", "Data Types", "Missing Values"
        ])
        
        with inspect_tab1:
            st.caption("First 5 rows of the raw dataset")
            st.dataframe(st.session_state.bikes_paris.head(), use_container_width=True)
        
        with inspect_tab2:
            st.caption("Detailed information about each column")
            col_info = pd.DataFrame({
                'Column Name': st.session_state.bikes_paris.columns,
                'Non-Null Count': st.session_state.bikes_paris.notna().sum(),
                'Null Count': st.session_state.bikes_paris.isnull().sum(),
                'Null Percentage': (st.session_state.bikes_paris.isnull().sum() / len(st.session_state.bikes_paris) * 100).round(2)
            })
            st.dataframe(col_info, use_container_width=True)
        
        with inspect_tab3:
            st.caption("Data types of each column")
            
            
            dtype_df = pd.DataFrame({
                'Column': st.session_state.bikes_paris.dtypes.index,
                'Data Type': st.session_state.bikes_paris.dtypes.values.astype(str)
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        with inspect_tab4:
            st.caption("Columns with missing values")
            missing_data = st.session_state.bikes_paris.isnull().sum()
            missing_data = missing_data[missing_data > 0].reset_index()
            
            if not missing_data.empty:
                missing_data.columns = ['Column', 'Missing Count']
                missing_data['Missing Percentage'] = (missing_data['Missing Count'] / len(st.session_state.bikes_paris) * 100).round(2)
                
                st.dataframe(missing_data, use_container_width=True)
            else:
                st.success("No missing values found in the dataset!")
    else:
        st.warning("Raw dataset is not available. Please check file paths and data loading process.")

def preprocessing_steps():
    st.subheader("Preprocessing Workflow")
    
    
    st.markdown("""
    This section explains the data preprocessing steps applied to transform the raw
    bike counter data into an analysis-ready dataset.
    """)
    
    # Workflow diagram
    workflow_steps = [
        {"step": "Data Loading", "color": "#3498db"},
        {"step": "Missing Value Handling", "color": "#2ecc71"},
        {"step": "Data Type Conversion", "color": "#9b59b6"},
        {"step": "Feature Engineering", "color": "#e67e22"},
        {"step": "Outlier Detection", "color": "#e74c3c"},
        {"step": "Data Validation", "color": "#1abc9c"}
    ]
    
   
    cols = st.columns(len(workflow_steps))
    for i, (col, step) in enumerate(zip(cols, workflow_steps)):
        with col:
            st.markdown(f"""
            <div style="
                background-color: {step['color']}; 
                padding: 10px; 
                border-radius: 5px; 
                color: white; 
                text-align: center;
                margin: 5px;
            ">
                <div style="font-size: 12px;">{i+1}. {step['step']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    
    with st.expander("1. Data Loading", expanded=True):
        st.markdown("""
        ### Data Loading
        
        The dataset was loaded from a CSV file with specific parameters to handle the file format correctly:
        
        - **Separator**: Semicolon (;) was used instead of the default comma
        - **Encoding**: UTF-8 encoding to properly handle French characters
        """)

    
    with st.expander("2. Missing Value Handling"):
        st.markdown("""
        ### Missing Value Handling
        
        We identified columns with missing values and applied appropriate strategies for each:
        
        - **Counters with missing IDs**: Removed rows where the counter ID was missing (essential identifier)
        - **Missing timestamps**: Removed records with missing timestamps as they couldn't be analyzed temporally
        - **Missing coordinates**: Kept records with missing coordinates for temporal analysis but excluded them from spatial analysis
        """)
    
    with st.expander("3. Data Type Conversion"):
        st.markdown("""
        ### Data Type Conversion
        
        We converted columns to their appropriate data types for analysis:
        
        - **Date & Time**: Converted string timestamps to datetime objects
        - **Numerical Values**: Ensured count data was numeric (integer)
        - **Coordinates**: Split coordinate strings into float latitude and longitude
        """)
        
       
        st.code("""
        # Convert date-time
        df['Cntg_Date_Time'] = pd.to_datetime(df['Cntg_Date_Time'])
        
        # Convert count to numeric
        df['Hrly_Cnt'] = pd.to_numeric(df['Hrly_Cnt'], errors='coerce')
        
        # Convert coordinates
        df[['latitude', 'longitude']] = df['Geo_Coord'].str.split(',', expand=True).astype(float)
        """, language="python")
        
        
        type_changes = pd.DataFrame({
            'Column': ['Cntg_Date_Time', 'Hrly_Cnt', 'Geo_Coord'],
            'Original Type': ['object (string)', 'object (string)', 'object (string)'],
            'Converted Type': ['datetime64[ns]', 'int64', 'Split into float lat/long']
        })
        
        st.table(type_changes)
    
    with st.expander("4. Feature Engineering"):
        st.markdown("""
        ### Feature Engineering
        
        We extracted and created new features to enrich our analysis capabilities:
        """)
        
        
        fe_tab1, fe_tab2, fe_tab3 = st.tabs(["Temporal Features", "Categorical Features", "Geospatial Features"])
        
        with fe_tab1:
            st.markdown("""
            #### Temporal Feature Extraction
            
            From the original `Cntg_Date_Time` timestamp, we extracted:
            
            - **Date Components**: Year, Month, Day
            - **Time Components**: Hour
            - **Calendar Features**: Day of week, Is weekend flag
            - **Time of Day Classification**: Morning, Afternoon, Evening, Night
            """)
            
            
            st.code("""
            # Extract date components
            df['Date'] = df['Cntg_Date_Time'].dt.date
            df['Time'] = df['Cntg_Date_Time'].dt.time
            df['Year'] = df['Cntg_Date_Time'].dt.year
            df['Month'] = df['Cntg_Date_Time'].dt.month
            df['Day'] = df['Cntg_Date_Time'].dt.day
            df['Hour'] = df['Cntg_Date_Time'].dt.hour
            
            # Extract day of week
            df['Weekday'] = df['Cntg_Date_Time'].dt.day_name()
            df['Weekday_Number'] = df['Cntg_Date_Time'].dt.dayofweek
            df['IsWeekend'] = df['Weekday_Number'].isin([5, 6])
            df['IsWeekend_name'] = df['IsWeekend'].map({True: 'Weekend', False: 'Weekday'})
            """, language="python")
        
        with fe_tab2:
            st.markdown("""
            #### Categorical Features
            
            We processed categorical features to make them suitable for analysis:
            
            - **Weekend Flag**: Created binary flag for weekend/weekday
            - **Time of Day**: Categorized hours into meaningful time periods
            """)
        
        with fe_tab3:
            st.markdown("""
            #### Geospatial Features
            
            We processed the coordinates to enable spatial analysis:
            
            - **Coordinate Splitting**: Split the combined coordinates into separate latitude and longitude
            - **Spatial Calculations**: Enabled distance calculations and spatial clustering
            """)
            
            
            st.code("""
            # Extract coordinates
            df[['latitude', 'longitude']] = df['Geo_Coord'].str.split(',', expand=True).astype(float)
            """, language="python")
    
    # Column Translation Section
    st.subheader("Column Translation (French to English)")
    st.markdown("""
    To make the dataset more accessible and consistent for analysis, we translated the original French column names to English.
    This improves code readability and ensures consistent column naming conventions.
    """)
    
    translation_dict = {
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
    
    # Create a DataFrame from the dictionary
    translation_df = pd.DataFrame({
        'Original (French)': translation_dict.keys(),
        'Translated (English)': translation_dict.values()
    })
    
    st.dataframe(translation_df, use_container_width=True)
   

def cleaned_dataset_exploration():
    
    if 'bikes_paris_clean' in st.session_state and st.session_state.bikes_paris_clean is not None:
        
        st.subheader("Cleaned Dataset Overview")
        
        # Basic statistics
        total_rows = st.session_state.bikes_paris_clean.shape[0]
        total_cols = st.session_state.bikes_paris_clean.shape[1]
        missing_values = st.session_state.bikes_paris_clean.isnull().sum().sum()
        missing_percentage = (missing_values / (total_rows * total_cols)) * 100
        
        # Show metrics with delta values compared to the original dataset
        if 'bikes_paris' in st.session_state and st.session_state.bikes_paris is not None:
            orig_rows = st.session_state.bikes_paris.shape[0]
            orig_cols = st.session_state.bikes_paris.shape[1]
            orig_missing = st.session_state.bikes_paris.isnull().sum().sum()
            
            row_delta = total_rows - orig_rows
            col_delta = total_cols - orig_cols
            missing_delta = missing_values - orig_missing
            
            # Create columns for metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{total_rows:,}", f"{row_delta:+,}")
            with col2:
                st.metric("Total Columns", total_cols, f"{col_delta:+}")
            with col3:
                st.metric("Missing Values", f"{missing_values:,}", f"{missing_delta:+,}")
        else:
            # Create columns for metrics without delta values
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{total_rows:,}")
            with col2:
                st.metric("Total Columns", total_cols)
            with col3:
                st.metric("Missing Values", f"{missing_values:,} ({missing_percentage:.2f}%)")
        
        
        st.subheader("Cleaned Dataset Inspection")
        
        
        inspect_tab1, inspect_tab2, inspect_tab3, inspect_tab4 = st.tabs([
            "Sample Data", "Data Types", "Missing Values", "Statistics"
        ])
        
        with inspect_tab1:
            st.caption("First 5 rows of the cleaned dataset")
            st.dataframe(st.session_state.bikes_paris_clean.head(), use_container_width=True)
                   
        with inspect_tab2:
            st.caption("Data types of each column in the cleaned dataset")
            
            
            dtype_df = pd.DataFrame({
                'Column': st.session_state.bikes_paris_clean.dtypes.index,
                'Data Type': st.session_state.bikes_paris_clean.dtypes.values.astype(str)
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        with inspect_tab3:
            st.caption("Columns with missing values in the cleaned dataset")
            missing_data = st.session_state.bikes_paris_clean.isnull().sum()
            missing_data = missing_data[missing_data > 0].reset_index()
            
            if not missing_data.empty:
                missing_data.columns = ['Column', 'Missing Count']
                missing_data['Missing Percentage'] = (missing_data['Missing Count'] / len(st.session_state.bikes_paris_clean) * 100).round(2)
                
                st.dataframe(missing_data, use_container_width=True)
                
                
                fig = px.bar(
                    missing_data,
                    x='Column',
                    y='Missing Count',
                    title='Missing Value Count by Column in Cleaned Dataset',
                    color='Missing Percentage',
                    color_continuous_scale='Greens',
                    text='Missing Percentage'
                )
                fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values found in the cleaned dataset!")
        
        with inspect_tab4:
            st.caption("Descriptive statistics of numeric columns in the cleaned dataset")
            st.dataframe(format_dataframe(st.session_state.bikes_paris_clean.describe()), use_container_width=True)

    else:
        st.warning("Cleaned dataset is not available. Please check file paths and data loading process.")