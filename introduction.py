import streamlit as st

def show_introduction():
    st.title("Pydaling Paris")
    st.header("Introduction")
    
    st.write("Paris stands at the forefront of urban mobility transformation, implementing ambitious initiatives to promote cycling as a sustainable transportation alternative. These efforts aim to reduce carbon emissions, improve air quality, and enhance quality of life for both residents and visitors. Central to this vision is the expansion of dedicated cycling infrastructure, including an extensive network of bike lanes supported by data-driven urban planning.")
    
    st.write("To measure the impact of these investments and better understand cycling patterns, the city has deployed approximately 108 bike counting stations throughout Paris. These stations continuously collect valuable data, recording the number of bicycles passing through on an hourly basis. Our analysis focuses on this comprehensive dataset, spanning from early 2023 through January 2025, providing insights into the evolving cycling behaviors across the city.")
    
    # Project goals in an expandable section
    with st.expander("Project Goals", expanded=True):
        st.subheader("Our Key Objectives")
        st.markdown("""
        This project addresses two primary objectives:
            
        1. **Pattern Identification**: We identify meaningful patterns in Parisian cycling behavior, 
        examining temporal trends and spatial variations. By analyzing these patterns, 
        we better understand how factors such as time of day, day of week, seasons, 
        and geographical location influence cycling activity throughout the city.
        
        2. **Predictive Modeling**: We develop machine learning models to forecast 
        future cycling patterns. These predictions can inform infrastructure planning, 
        resource allocation, and policy decisions to further support and enhance 
        Paris's cycling initiative.
        """)
    
    st.subheader("Analysis Focus")
    st.markdown("""
    Our analysis specifically examines:
    * Temporal patterns, with particular attention to six critical time points throughout the day
    * Spatial distribution of cycling traffic across different counter stations and neighborhoods
    """)
    
    st.write("The findings presented in this report offer valuable insights for urban planners, policymakers, and transportation authorities seeking to foster sustainable mobility in Paris and potentially serve as a model for other cities pursuing similar goals.")