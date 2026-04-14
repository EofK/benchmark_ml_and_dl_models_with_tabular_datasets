import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Setup the Page Layout
st.set_page_config(page_title="DL Model Deployment Score", layout="wide")
st.title("Interactive DL Model Deployment Score")
st.markdown("Adjust the weights below to see how model rankings change based on deployment priorities.")

# 2. Embed the Cleaned Data
data = {
    "model": [
        "FT-Transformer", "TabTransformer", "SAINT", "Self-Attention MLP", 
        "TabFastFormer", "GANDALF", "GATE", "NODE", "AutoInt", "DANet", 
        "Category Embedding", "TabMLP"
    ],
    "family": [
        "Transformer", "Transformer", "Transformer", "Transformer", 
        "Transformer", "Tree-Neural", "Tree-Neural", "Tree-Neural", 
        "Feature-Interaction", "Feature-Interaction", "MLP", "MLP"
    ],
    "accuracy_norm": [0.95, 0.00, 0.49, 0.87, 1.00, 0.83, 0.62, 0.47, 0.97, 0.40, 0.99, 1.00],
    "throughput_norm": [0.10, 0.88, 0.49, 0.62, 0.48, 0.74, 0.06, 0.00, 0.52, 0.15, 1.00, 0.81]
}
df = pd.DataFrame(data)

# 3. Define the Global Colors
# family_colors = {
#     "Transformer":         "#C9B4D0",  # muted lavender
#     "Tree-Neural":         "#B4D1D0",  # muted teal-gray
#     "Feature-Interaction": "#E8D0D0",  # soft rose
#     "MLP":                 "#D0C4B4",  # warm sand
# }

family_colors = {
    "Transformer":         "#9A7BA5",  # Rich lavender
    "Tree-Neural":         "#7A9F9E",  # Rich teal-gray
    "Feature-Interaction": "#C19A9A",  # Rich rose
    "MLP":                 "#A3937F",  # Rich sand
}

# 4. Create the Interactive Slider
acc_weight = st.slider(
    "Accuracy Weight (%)", 
    min_value=0, 
    max_value=100, 
    value=60, 
    step=1
)
thru_weight = 100 - acc_weight

st.markdown(f"**Current Weights:** Accuracy = {acc_weight}% | Throughput = {thru_weight}%")

# 5. Calculate the Dynamic Deployment Score
df["Deployment Score"] = (df["accuracy_norm"] * (acc_weight / 100.0)) + (df["throughput_norm"] * (thru_weight / 100.0))

# Sort the dataframe so the highest score appears at the top of the chart
df = df.sort_values(by="Deployment Score", ascending=True)

# 6. Build the Chart
fig = px.bar(
    df, 
    x="Deployment Score", 
    y="model", 
    color="family",
    color_discrete_map=family_colors,
    orientation="h",
    text="Deployment Score",
    height=600,
    category_orders={"model": df["model"].tolist()}  # <--- THIS FORCES THE DYNAMIC SORT
)

# Format the text labels and layout
fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig.update_layout(
    xaxis_title="Deployment Score (0.0 to 1.0)",
    yaxis_title="Deep Learning Model",
    xaxis=dict(range=[0, 1.1]), 
    plot_bgcolor="white",
    
    # --- NEW LEGEND SIZE CONTROLS ---
    legend_title=dict(text="Model Family", font=dict(size=19)), # Tweak this number (default is ~13)
    legend=dict(
        font=dict(size=15), # Tweak this number (default is ~12)
        itemsizing="constant" # Keeps the color boxes nicely proportioned
    )
)

# 7. Render the Chart
st.plotly_chart(fig, use_container_width=True)

# cd notebooks
# streamlit run benchmark_dashboard.py