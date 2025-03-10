import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

def render_conclusion_page():

    st.header("Conclusion")
    st.write("The main objective for our project was to determine the most suitable model for predicting the hourly bike counts for the city of Paris. We compared several models, such as the Linear Regression (LR), Random Forest (RF), Decision Trees (DT), and additionally the K-Means and DBSCAN methods. In the charts below, we visualized an overview of the most important model outcomes.")


    # Create a DataFrame
    data = {
        "Model": ["LR (Geo)", "LR (Non-Geo)", "RF (Geo)", "RF (Non-Geo)", "DT (Geo)", "DT (Non-Geo)"],
        "MAE": [56.59, 56.01, 21.62, 20.81, 24.57, 31.38],
        "RMSE": [92.28, 92.07, 43.5, 44, 50.26, 59.64],
        "Train R²-Score": [0.203, 0.202, 0.875, 0.896, 0.877, 0.902],
        "Test R²-Score": [0.206, 0.203, 0.824, 0.818, 0.764, 0.665],
    }

    df = pd.DataFrame(data)

    
    def highlight_best(s):
        """Highlight the best model row in blue."""
        return ['background-color: #AFCBF5; font-weight: bold' if s.name == 2 else '' for _ in s]

    styled_df = (
        df.style
        .apply(highlight_best, axis=1)  
        .set_properties(**{"text-align": "center"})  
        .format(precision=2) 
    )

    
    st.title("Model Performance Comparison")
    st.dataframe(styled_df, hide_index=True, use_container_width=True)

    
    st.subheader("R² Score Comparison")

    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df["Model"], df["Test R²-Score"], color="#1f497d", width=0.6)

    
    for i, v in enumerate(df["Test R²-Score"]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=12, fontweight='bold')

    
    ax.set_xlabel("Model")
    ax.set_ylabel("R² Score")
    ax.set_title("R² Score Comparison", fontsize=14)
    ax.set_ylim(0, 0.9)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    
    st.pyplot(fig)

    st.write("""
On the other hand, Silhouette Coefficient, Davies-Bouldin Index, Calinski-Harabasz Index, and Dunn Index cannot be evaluated using R² Score comparison because they measure cluster quality rather than predictive accuracy. Unlike R² Score, which evaluates the goodness of fit in regression models, these indices assess cohesion and separation of clusters. Therefore, they are not comparable, as agreed upon by multiple team members during the decision-making process.

We will sum up some of the most important takeaways from the model comparison:
- Adding **geographical data** to the models only minimally changes the outcome of the models.
- The **Random Forest with Geographical data** seems to be the most accurate and robust model for predicting the Hourly Bike Counts, showing high predictive power and good generalization.
- The **Decision Tree** models perform well, but they are showing slight overfitting.
- The marginal impact of longitude and latitude suggests that bike counts are influenced more by temporal patterns than location.
- **Non-linear models** are essential for capturing the complex patterns present in this dataset. This is clearly visible since the **Linear Regression models** perform poorly in comparison to the others.
- In addition, **HeatMap** traffic visualization adds a visual representation of cyclist behavior patterns, but **cannot** be used as a stand-alone metric for making a final decision.

Based on the model evaluation and comparison, the Random Forest model with Geographic Data was selected as the best-performing model. This decision is supported by its highest R² Score (0.824) and consistent generalization to unseen data, demonstrating its robustness and predictive power.

During the analysis, it was observed that hourly counts varied significantly with time of day and location. The feature importance results from the selected model validate these observations, highlighting that hour, longitude, and latitude are the most influential predictors. This correlation between observed patterns and model feature importance enhances the model's interpretability and supports its accuracy in predicting hourly bike counts.
""")