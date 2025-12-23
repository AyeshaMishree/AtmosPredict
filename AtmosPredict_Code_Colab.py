# -------------------------------
# ATMOST-PREDICT
# -------------------------------

!pip install streamlit pandas matplotlib seaborn scipy scikit-learn pyngrok
!pip install pyngrok

from pyngrok import ngrok
ngrok.set_auth_token("paste_your_ngrok_token_here")

app_code = """
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import zscore
from scipy.interpolate import CubicSpline
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose

# -------------------------------
# Page Title
# -------------------------------
st.markdown("<h1 style='text-align: center; color: lightblue;'>Weather Data Analysis</h1>", unsafe_allow_html=True)

# -------------------------------
# Preloaded Datasets
# -------------------------------
cities = {
    "Karachi": "/content/Karachi_weather.xlsx",
    "Lahore": "/content/Lahore_weather.xlsx",
    "Islamabad": "/content/Islamabad_weather.xlsx",
    "Multan": "/content/Multan_weather.xlsx",
    "Faisalabad": "/content/Faisalabad_weather.xlsx",
    "Rawalpindi": "/content/Rawalpindi_weather.xlsx"
}

# -------------------------------
# Sidebar - User Selection
# -------------------------------
st.sidebar.header("Select Options")
selected_city = st.sidebar.selectbox("Select City", options=list(cities.keys()))

# Load selected city's dataset
weather_data = pd.read_excel(cities[selected_city])
weather_data["datetime"] = pd.to_datetime(weather_data["datetime"])

# -------------------------------
# Date Picker Filter
# -------------------------------
min_date = weather_data["datetime"].min()
max_date = weather_data["datetime"].max()

with st.sidebar.container():
    date_range = st.sidebar.date_input(
        f"Select Date Range for {selected_city}",
        value=(min_date.date(), max_date.date()),
        min_value=min_date.date(),
        max_value=max_date.date()
    )

# -------------------------------
# Date Range Validation
# -------------------------------
if not isinstance(date_range, (list, tuple)) or len(date_range) != 2:
    st.sidebar.warning("Please select BOTH start and end dates.")
    st.stop()

start_date, end_date = date_range

if start_date is None or end_date is None:
    st.sidebar.warning("Please select BOTH start and end dates.")
    st.stop()

# -------------------------------
# Safe filtering only after valid selection
# -------------------------------
filtered_data = weather_data[
    (weather_data["datetime"].dt.date >= start_date) &
    (weather_data["datetime"].dt.date <= end_date)
]

st.write(f"### {selected_city} Original Weather Data Preview")
st.dataframe(filtered_data)

# -------------------------------
# Columns to Remove & Cleaning
# -------------------------------
columns_to_remove = [
    'feelslikemax', 'precipcover', 'snowdepth', 'feelslikemin',
    'preciptype', 'feelslike', 'precip', 'description',
    'precipprob', 'icon', 'snow'
]

weather_data_cleaned = filtered_data.drop(columns=columns_to_remove, errors="ignore")
weather_data_cleaned["Day_of_Year"] = weather_data_cleaned["datetime"].dt.dayofyear
weather_data_cleaned["Month"] = weather_data_cleaned["datetime"].dt.month
weather_data_cleaned = weather_data_cleaned.ffill().bfill()
if weather_data_cleaned.duplicated().any():
    weather_data_cleaned = weather_data_cleaned.drop_duplicates()

# -------------------------------
# Duplicate Check
# -------------------------------
if weather_data_cleaned.duplicated().any():
    weather_data_cleaned = weather_data_cleaned.drop_duplicates()
    st.write(" Duplicates removed.")
else:
    st.write(" No duplicates found.")

# -------------------------------
# Data After Cleaning Preview
# -------------------------------
st.write("### Data After Cleaning")
st.dataframe(weather_data_cleaned)


# -------------------------------
# Lagrange Interpolation (Numerical Method)
# -------------------------------
st.write("## Lagrange Interpolation Prediction")

# Take a small sample to avoid instability
lagrange_data = weather_data_cleaned.sort_values("datetime").tail(7)

x = np.arange(len(lagrange_data))
y = lagrange_data["temp"].values

def lagrange_interpolation(x, y, x_new):
    result = 0
    n = len(x)
    for i in range(n):
        term = y[i]
        for j in range(n):
            if i != j:
                term *= (x_new - x[j]) / (x[i] - x[j])
        result += term
    return result

# Predict next day's temperature
next_day_index = len(x)
lagrange_pred = lagrange_interpolation(x, y, next_day_index)

st.success(f"Lagrange Predicted Temperature (Next Day): {lagrange_pred:.2f} °C")


# -------------------------------
# Newton’s Divided Difference Prediction
# -------------------------------
st.write("## Newton’s Divided Difference Prediction")

# Use last 5 temperature points
recent_days = lagrange_data.index.values  # use same x as Lagrange
recent_temps = lagrange_data["temp"].values

# Newton’s Divided Difference coefficients
def newton_divided_diff(x, y):
    n = len(y)
    coef = np.copy(y).astype(float)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (x[j:n] - x[0:n-j])
    return coef

# Newton prediction
def newton_predict(x_data, coef, x):
    n = len(coef) - 1
    result = coef[n]
    for k in range(n-1, -1, -1):
        result = result * (x - x_data[k]) + coef[k]
    return result

# Calculate coefficients
coef = newton_divided_diff(recent_days, recent_temps)
next_day = recent_days[-1] + 1
newton_pred = newton_predict(recent_days, coef, next_day)

st.success(f"Newton Predicted Temperature (Next Day): {newton_pred:.2f} °C")


# -------------------------------
# RMSE for Numerical Predictions
# -------------------------------
import numpy as np

# For demonstration, using last 7 actual temps vs predicted next day
actual_next_day = np.array([weather_data_cleaned["temp"].iloc[-1]])  # actual last day
pred_next_day = np.array([lagrange_pred])  # predicted by Lagrange (same as Newton here)

rmse = np.sqrt(np.mean((actual_next_day - pred_next_day)**2))
st.write(f"### RMSE of Numerical Prediction: {rmse:.2f} °C")

# -------------------------------
# Numerical vs ML Comparison Table
# -------------------------------

# Prepare ML prediction RMSE for comparison
ml_data = weather_data_cleaned.select_dtypes(include=[np.number])
if "temp" in ml_data.columns and len(ml_data) >= 10:
    X = ml_data.drop("temp", axis=1)
    y = ml_data["temp"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
    ml_model.fit(X_train, y_train)
    ml_preds = ml_model.predict(X_test)
    ml_rmse = np.sqrt(mean_squared_error(y_test, ml_preds))
else:
    ml_rmse = np.nan

# Create a comparison DataFrame
comparison_df = pd.DataFrame({
    "Method": ["Lagrange Interpolation", "Newton Divided Difference", "Random Forest ML"],
    "Predicted Next Day Temp (°C)": [lagrange_pred, newton_pred, ml_preds[-1] if "ml_preds" in locals() else np.nan],
    "RMSE (°C)": [rmse, rmse, ml_rmse]
})

st.write("### Numerical vs Machine Learning Comparison")
st.dataframe(comparison_df)


numeric_cols = weather_data_cleaned.select_dtypes(include=["number"])
z_scores = numeric_cols.apply(zscore)
weather_data_cleaned = weather_data_cleaned[(z_scores.abs() < 3.5).all(axis=1)]

# -------------------------------
# Seasonal Classification
# -------------------------------
weather_data_cleaned["Season"] = weather_data_cleaned["Month"].map({
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn"
})

# -------------------------------
# Function to plot transparent Matplotlib
# -------------------------------
def plot_transparent(fig):
    fig.patch.set_alpha(0)
    for ax in fig.axes:
        ax.set_facecolor("none")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
    st.pyplot(fig, transparent=True)

# -------------------------------
# ALL GRAPHS START HERE
# -------------------------------

# 1 Correlation Heatmap
st.write("### Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(weather_data_cleaned.select_dtypes(include=["number"]).corr(),
            annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
plot_transparent(fig)

# 2 Monthly Average Temp Trend
monthly_avg = weather_data_cleaned.groupby("Month")["temp"].mean()
st.write("### Monthly Average Temperature Trend")
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(monthly_avg.index, monthly_avg.values, marker="o")
plot_transparent(fig)

# 3 Yearly Temp Trend
yearly_avg = weather_data_cleaned.groupby(weather_data_cleaned["datetime"].dt.year)["temp"].mean()
st.write("### Yearly Temperature Trend")
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(yearly_avg.index, yearly_avg.values, marker="o")
plot_transparent(fig)

# 4 Temp Histogram
st.write("### Temperature Histogram")
fig, ax = plt.subplots(figsize=(10,5))
ax.hist(weather_data_cleaned["temp"], bins=15, edgecolor="black")
plot_transparent(fig)

# 5 Temp vs Humidity
if "humidity" in weather_data_cleaned.columns:
    st.write("### Temperature vs Humidity")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(weather_data_cleaned["humidity"], weather_data_cleaned["temp"])
    ax.set_xlabel("Humidity")
    ax.set_ylabel("Temperature")
    plot_transparent(fig)

# 6 Wind Speed Distribution
if "windspeed" in weather_data_cleaned.columns:
    st.write("### Wind Speed Distribution")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(weather_data_cleaned["windspeed"], bins=15)
    plot_transparent(fig)

# 7 Season-wise Temp Distribution
st.write("### Season-wise Temperature Distribution")
fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(x="Season", y="temp", data=weather_data_cleaned, ax=ax)
plot_transparent(fig)

# 8 Daily Temp Trend
st.write("### Daily Temperature Trend")
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(weather_data_cleaned["datetime"], weather_data_cleaned["temp"])
ax.set_xlabel("Date")
ax.set_ylabel("Temperature")
plot_transparent(fig)

# 9 7-Day Rolling Avg
st.write("### 7-Day Rolling Average Temperature")
weather_data_cleaned["Rolling_Temp"] = weather_data_cleaned["temp"].rolling(7).mean()
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(weather_data_cleaned["datetime"], weather_data_cleaned["Rolling_Temp"])
ax.set_xlabel("Date")
ax.set_ylabel("Temperature")
plot_transparent(fig)

# 10 Temp vs Pressure
if "pressure" in weather_data_cleaned.columns:
    st.write("### Temperature vs Pressure")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(weather_data_cleaned["pressure"], weather_data_cleaned["temp"])
    ax.set_xlabel("Pressure")
    ax.set_ylabel("Temperature")
    plot_transparent(fig)

# 11 Precip vs Temp
if "precip" in weather_data_cleaned.columns:
    st.write("### Precipitation vs Temperature")
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(weather_data_cleaned["precip"], weather_data_cleaned["temp"])
    ax.set_xlabel("Precipitation")
    ax.set_ylabel("Temperature")
    plot_transparent(fig)

# 12 Monthly Temp Box Plot
st.write("### Monthly Temperature Distribution")
fig, ax = plt.subplots(figsize=(10,5))
sns.boxplot(x="Month", y="temp", data=weather_data_cleaned, ax=ax)
plot_transparent(fig)

# 13 Temp Anomaly Detection
st.write("### Temperature Anomaly Detection")
weather_data_cleaned["Temp_Zscore"] = zscore(weather_data_cleaned["temp"])
anomalies = weather_data_cleaned[weather_data_cleaned["Temp_Zscore"].abs() > 3]
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(weather_data_cleaned["datetime"], weather_data_cleaned["temp"], label="Normal")
ax.scatter(anomalies["datetime"], anomalies["temp"], color="red", label="Anomaly")
ax.legend()
plot_transparent(fig)

# 14 Temp vs Day of Year
st.write("### Temperature vs Day of Year")
fig, ax = plt.subplots(figsize=(12,5))
ax.scatter(weather_data_cleaned["Day_of_Year"], weather_data_cleaned["temp"], alpha=0.5)
plot_transparent(fig)

# 15 Smoothed Temp (Cubic Spline)
st.write("### Smoothed Temperature Trend (Cubic Spline)")
sorted_data = weather_data_cleaned.sort_values("Day_of_Year")
cs = CubicSpline(sorted_data["Day_of_Year"], sorted_data["temp"])
x_new = np.linspace(1,365,365)
y_new = cs(x_new)
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(x_new, y_new)
plot_transparent(fig)

# 16 Humidity by Season
if "humidity" in weather_data_cleaned.columns:
    st.write("### Humidity Distribution by Season")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.boxplot(x="Season", y="humidity", data=weather_data_cleaned, ax=ax)
    plot_transparent(fig)

# 17 Monthly Temp Variability
st.write("### Monthly Temperature Variability")
monthly_std = weather_data_cleaned.groupby("Month")["temp"].std()
fig, ax = plt.subplots(figsize=(10,5))
ax.bar(monthly_std.index, monthly_std.values)
plot_transparent(fig)

# 18 Pairplot
st.write("### Pairwise Feature Relationships")
pair_fig = sns.pairplot(weather_data_cleaned.select_dtypes(include="number").iloc[:,:6])
plot_transparent(pair_fig.fig)

# 19 Lag Plot
st.write("### Temperature Lag Plot")
fig, ax = plt.subplots(figsize=(6,6))
ax.scatter(weather_data_cleaned["temp"].shift(1), weather_data_cleaned["temp"], alpha=0.5)
plot_transparent(fig)

# 20 Cumulative Temp Trend
st.write("### Cumulative Temperature Trend")
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(weather_data_cleaned["datetime"], weather_data_cleaned["temp"].cumsum())
plot_transparent(fig)

# 21 Plotly Interactive Daily Trend
st.write("## Interactive Daily Temperature Trend")
fig = px.line(weather_data_cleaned, x="datetime", y="temp", title="Daily Temperature (Interactive)")
st.plotly_chart(fig, use_container_width=True)

# 22 Plotly Temp vs Humidity Interactive
if "humidity" in weather_data_cleaned.columns:
    st.write("## Interactive Temperature vs Humidity")
    fig = px.scatter(weather_data_cleaned, x="humidity", y="temp", color="Season",
                     title="Temperature vs Humidity (Interactive)")
    st.plotly_chart(fig, use_container_width=True)

# 23 Future 30-Day Prediction
st.write("## Future 30-Day Temperature Prediction")
weather_data_cleaned = weather_data_cleaned.sort_values("datetime")
weather_data_cleaned["Temp_Lag1"] = weather_data_cleaned["temp"].shift(1)
weather_data_cleaned["Temp_Lag7"] = weather_data_cleaned["temp"].shift(7)
forecast_data = weather_data_cleaned.dropna()
X = forecast_data[["Day_of_Year","Temp_Lag1","Temp_Lag7"]]
y = forecast_data["temp"]
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)
last_date = weather_data_cleaned["datetime"].max()
future_dates = pd.date_range(start=last_date+pd.Timedelta(days=1), periods=30)
future_df = pd.DataFrame({
    "datetime": future_dates,
    "Day_of_Year": future_dates.dayofyear,
    "Temp_Lag1": weather_data_cleaned["temp"].iloc[-1],
    "Temp_Lag7": weather_data_cleaned["temp"].iloc[-1]
})
future_df["Predicted_Temp"] = model.predict(future_df[["Day_of_Year","Temp_Lag1","Temp_Lag7"]])
fig = go.Figure()
fig.add_trace(go.Scatter(x=weather_data_cleaned["datetime"], y=weather_data_cleaned["temp"], mode="lines", name="Historical"))
fig.add_trace(go.Scatter(x=future_df["datetime"], y=future_df["Predicted_Temp"], mode="lines", name="Predicted"))
fig.update_layout(title="Future Temperature Forecast (30 Days)", xaxis_title="Date", yaxis_title="Temperature")
st.plotly_chart(fig, use_container_width=True)

# 24 Trend & Seasonal Decomposition
st.write("## Trend and Seasonal Decomposition")
ts_data = weather_data_cleaned.set_index("datetime")["temp"].asfreq("D").interpolate()
decomposition = seasonal_decompose(ts_data, model="additive", period=30)
fig = go.Figure()
fig.add_trace(go.Scatter(x=ts_data.index, y=decomposition.trend, name="Trend"))
fig.add_trace(go.Scatter(x=ts_data.index, y=decomposition.seasonal, name="Seasonal"))
fig.add_trace(go.Scatter(x=ts_data.index, y=decomposition.resid, name="Residual"))
fig.update_layout(title="Temperature Decomposition (Trend + Seasonal + Residual)", xaxis_title="Date")
st.plotly_chart(fig, use_container_width=True)

# 25 ML Temperature Prediction
st.write("### Machine Learning Temperature Prediction")
ml_data = weather_data_cleaned.select_dtypes(include=[np.number])
if "temp" in ml_data.columns and len(ml_data) >= 10:
    X = ml_data.drop("temp", axis=1)
    y = ml_data["temp"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    st.success("Model Performance")
    st.write(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
    st.write(f"MSE: {mean_squared_error(y_test, preds):.2f}")


# -----------------------------------------
# Real-Time Weather via OpenWeatherMap API (Hybrid Prediction)
# -----------------------------------------
import requests

st.write("## Real-Time Weather Validation (Hybrid)")

# Your API key
OWM_API_KEY = "paste_your_open_weather_map_api_here"

# Let user select city
realtime_city = st.selectbox("Select City for Real-Time Data", list(cities.keys()))

# Function to fetch real-time temperature
def get_real_time_temp(city_name, api_key):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city_name}&units=metric&appid={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data["main"]["temp"]
        else:
            st.error(f"Failed to fetch data. Status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Button to fetch real-time temp and calculate hybrid prediction
if st.button("Fetch Real-Time & Hybrid Prediction"):
    real_temp = get_real_time_temp(realtime_city, OWM_API_KEY)

    if real_temp is not None:
        st.success(f"Current Temperature in {realtime_city}: {real_temp:.2f} °C")

        # 1 ML prediction
        ml_pred_next_day = preds[-1] if "preds" in locals() else np.nan

        # 2 Numerical prediction (average of Lagrange and Newton)
        numerical_pred = (lagrange_pred + newton_pred) / 2

        # 3 Weighted Hybrid Prediction
        hybrid_pred = 0.4 * numerical_pred + 0.4 * ml_pred_next_day + 0.2 * real_temp

        st.write(f"Hybrid Predicted Temperature (Next Day): {hybrid_pred:.2f} °C")
        st.write(f"Difference from Real-Time Temp: {abs(hybrid_pred - real_temp):.2f} °C")
    else:
        st.error("Could not fetch real-time temperature.")

"""

with open("app.py", "w", encoding="utf-8") as f:
    f.write(app_code)

# Run Streamlit + ngrok tunnel
from pyngrok import ngrok
import os, time

# Kill old tunnels
ngrok.kill()

# Start Streamlit in background
os.system("streamlit run app.py --server.port 8501 --server.headless true &")

# Wait for Streamlit to start
time.sleep(10)

# Start ngrok tunnel
public_url = ngrok.connect(8501)
print("OPEN THIS LINK:", public_url)