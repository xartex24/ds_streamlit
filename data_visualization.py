import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image

def data_visualization_page():
    
    st.header("📊 Data Visualization")
    st.write("Down below, we can see both the initial temporal and demographical analysis, creating broad overview of the dataset")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📉 Temporal Analysis", "📍 Demographical Analysis"])

   
    if 'bikes_paris_clean' in st.session_state and st.session_state.bikes_paris_clean is not None:
       
        df = st.session_state.bikes_paris_clean.copy()
        
        # ========================== Hourly Trends & Seasonality Analysis ==========================
        with tab1:
            st.write("In this section we will visualize the temporal data regarding the hourly counts with focus on several different parameters.")

            
            if "Year" in df.columns:
                df["Year"] = df["Year"].astype(int)

                              
                min_year, max_year = int(df["Year"].min()), int(df["Year"].max())

                
                all_years = st.checkbox("Show all years", value=True)

                if all_years:
                    filtered_df = df.copy()
                    selected_year = "Total"
                else:
                    # Use a slider instead of a dropdown for year selection
                    selected_year = st.slider("Select Year", min_year, max_year, min_year)
                    filtered_df = df[df["Year"] == selected_year]

                # Average Hourly Count per Weekday
                hourly_avg = filtered_df.groupby(["hour", "Weekday"])["Hrly_Cnt"].mean().reset_index()

                # Average Hourly Count by Time of Day
                time_of_day_avg = filtered_df.groupby("Time_Of_Day")["Hrly_Cnt"].mean().reset_index()

                # Weekly Average for Seasonality Trend
                if "Date" in filtered_df.columns:
                    # Ensure Date column is datetime
                    if not pd.api.types.is_datetime64_any_dtype(filtered_df["Date"]):
                        filtered_df["Date"] = pd.to_datetime(filtered_df["Date"])
                    
                    weekly_avg = filtered_df.resample("W", on="Date")["Hrly_Cnt"].mean().reset_index()

                   
                    time_of_day_labels = ["Early morning", "Morning", "Middle of the day", "Afternoon", "Evening", "Night"]

                    # Line chart
                    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

                    # Create the plot with ordered hue
                    fig1 = plt.figure(figsize=(10, 6))
                    sns.lineplot(data=hourly_avg, x="hour", y="Hrly_Cnt", hue="Weekday", hue_order=weekday_order,palette="cubehelix")
                    plt.title(f"Avg. Hourly Bicycle Counts - {selected_year}", fontsize=14)
                    plt.xlabel("Hour", fontsize=12)
                    plt.ylabel("Avg. Count", fontsize=12)
                    plt.legend(fontsize=10, loc="upper right", frameon=False)
                    plt.tight_layout()
                    st.pyplot(fig1)

                    # Bar chart 
                    fig2 = plt.figure(figsize=(10, 6))
                    sns.barplot(data=time_of_day_avg, x="Time_Of_Day", y="Hrly_Cnt", palette="cubehelix")
                    plt.title("Avg. Hourly Count by Time of Day", fontsize=14)
                    plt.xlabel("Time of Day", fontsize=12)
                    plt.ylabel("Avg. Count", fontsize=12)
                    plt.xticks(range(len(time_of_day_labels)), time_of_day_labels, rotation=30, ha="right", fontsize=10)
                    plt.tight_layout()
                    st.pyplot(fig2)

                    # Scatter plot
                    
                    fig3 = plt.figure(figsize=(12, 7))  
                    time_mapping = {i: label for i, label in enumerate(time_of_day_labels)}
                    sns.scatterplot(data=filtered_df, x="Time_Of_Day", y="Hrly_Cnt", hue="IsWeekend_name", alpha=0.5, palette="cubehelix")
                    plt.title("Weekend vs. Weekday Distribution", fontsize=16, pad=20)
                    plt.xlabel("Time of Day", fontsize=13, labelpad=15)
                    plt.ylabel("Hourly Count", fontsize=13, labelpad=15)
                    plt.xticks(sorted(filtered_df['Time_Of_Day'].unique()),time_of_day_labels, rotation=45, ha="right", fontsize=11)
                    plt.subplots_adjust(bottom=0.15)
                    plt.legend(fontsize=11,loc="upper right", framealpha=0.9, title="Day Type")

                    plt.tight_layout()
                    st.pyplot(fig3)

                    # Line chart
                    fig4 = plt.figure(figsize=(10, 6))
                    sns.lineplot(data=weekly_avg, x="Date", y="Hrly_Cnt", palette="cubehelix")
                    plt.fill_between(weekly_avg["Date"], weekly_avg["Hrly_Cnt"], alpha=0.2, color="gray") 
                    plt.title("Average Weekly Bicycle Count", fontsize=14)
                    plt.xlabel("Date", fontsize=12)
                    plt.ylabel("Count", fontsize=12)
                    plt.xticks(rotation=30)
                    plt.tight_layout()
                    st.pyplot(fig4) 

                    #  Observations
                    st.markdown("""
                    ### Key Insights:
                    - **High peaks in hourly biking counts** can be seen around hour 7 and 17, indicating work traffic.
                    - **Hourly counts distributed by time of day** further confirm this trend.
                    - **Weekends show more spread-out usage**, likely due to leisure or tourism.
                    - **Seasonal trends peak in summer and decline in winter.**
                    """)
                else:
                    st.error("Date column not found in the dataset")
            else:
                st.error("Year column not found in the dataset")

        # ========================== Demographical Analysis ==========================
        with tab2:
            st.write("In this section we will visualize the geographical data, comparing multiple counter sites and displaying them on a map.")

            
            if "Cntr_Name" in df.columns and "Hrly_Cnt" in df.columns:
                location_counts_df = df.groupby("Cntr_Name")["Hrly_Cnt"].sum().reset_index()
                location_counts_df.columns = ["Location", "Total_Hourly_Count"]

                location_counts_df["Total_Hourly_Count"].fillna(0, inplace=True)
                location_counts_df = location_counts_df[location_counts_df["Total_Hourly_Count"] > 0]

                top_10_locations = location_counts_df.sort_values(by="Total_Hourly_Count", ascending=False).drop_duplicates().head(10)

                # Set up Figure
                fig, ax = plt.subplots(figsize=(14, 10), dpi=300)

                # Horizontal Bar Chart
                sns.barplot(
                    data=top_10_locations, 
                    x="Total_Hourly_Count", 
                    y="Location", 
                    palette="cubehelix", 
                    ax=ax
                )

                
                ax.yaxis.set_label_position("right") 
                ax.yaxis.tick_right()

                
                ax.set_title("Top 10 Locations by Hourly Counts", fontsize=10)  
                ax.set_xlabel("Sum of Hourly Counts", fontsize=10)
                ax.set_ylabel("Location", fontsize=10)
                ax.tick_params(axis="both", labelsize=6)

                ax.get_xaxis().get_major_formatter().set_scientific(False)

                sns.despine(left=True, bottom=True)

                # Save as PNG
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=300)
                buf.seek(0)

                
                st.image(buf, caption="Top 10 Locations by Hourly Counts", use_column_width=True)
            else:
                st.error("Required columns (Cntr_Name, Hrly_Cnt) not found in the dataset")

            try:
                image_path = "03_Exports\map.jpg"  
                image = Image.open(image_path)

                st.image(image, caption="Cycling Counters & Tourist Attractions", width=1000)
            except FileNotFoundError:
                st.error("Map image file 'map.jpg' not found. Please ensure it exists in the current directory.")
            except Exception as e:
                st.error(f"Error loading map image: {e}")


            st.markdown("""
            ### Key Insights:
            - **Bike usage is concentrated in key locations**, with top areas having nearly twice the counts of lower-ranked locations.
            - **Traffic flow is primarily South-North and North-South**, influenced by commuting patterns and tourists moving toward the city center.
            - **High-traffic areas could benefit from improved cycling infrastructure**, such as expanded bike lanes, bike-sharing programs, or safety enhancements.
            """)
    else:
        st.warning("Cleaned dataset is not available. Please check file paths and data loading process.")