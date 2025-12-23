# Weather Data Analysis & Prediction System

This project is a **Streamlit-based interactive weather analysis and prediction system**.  
It analyzes historical weather data of major Pakistani cities and predicts temperature using **Numerical Analysis techniques**, **Machine Learning**, and **real-time weather APIs**.

<img width="1600" height="703" alt="image" src="https://github.com/user-attachments/assets/ab9b53d4-921c-4927-81fd-480a81d17b00" />

---

## Project Overview
The application allows users to:
- Select a city and date range
- Clean and preprocess weather data
- Analyze trends using statistical and visual techniques
- Predict future temperature using numerical methods and ML
- Validate predictions using real-time weather data

---

## Numerical Analysis (NA) Concepts Used

This project applies the following **Numerical Analysis techniques**:

- **Lagrange Interpolation**  
  Used to predict the **next day’s temperature** based on recent temperature values.

- **Newton’s Divided Difference Method**  
  Another polynomial interpolation method used for temperature prediction and comparison with Lagrange.

- **Cubic Spline Interpolation**  
  Used to generate a **smooth temperature trend curve** over the year.

- **RMSE (Root Mean Square Error)**  
  Used to measure the error between predicted and actual temperature values.

- **Z-Score Method**  
  Used for **outlier detection and anomaly identification** in temperature data.

---

## Machine Learning Concepts Used

- **Random Forest Regressor**
  - Used for temperature prediction
  - Used for future **30-day forecasting**
  - Compared against numerical methods using RMSE

- **Train-Test Split**
  - Dataset split into training and testing sets

- **Evaluation Metrics**
  - MAE (Mean Absolute Error)
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Square Error)

---

## Time Series & Statistical Analysis

- **Rolling Average (7-day)** for trend smoothing  
- **Seasonal Classification** (Winter, Spring, Summer, Autumn)  
- **Seasonal Decomposition**
  - Trend
  - Seasonal
  - Residual components

---

## Data Visualization

The project includes **25+ visualizations**, including:
- Correlation heatmap
- Daily, monthly, and yearly trends
- Histograms and box plots
- Anomaly detection plots
- Lag plots and cumulative trends
- Interactive Plotly graphs

---

## APIs Used

### 1. OpenWeatherMap API
- Fetches **real-time temperature data**
- Used to validate predictions
- Enables hybrid prediction

### 2. Ngrok API
- Creates a **public URL** for running the Streamlit app from Google Colab

---

## Hybrid Prediction Model

The final prediction combines:
- Numerical prediction (Lagrange + Newton)
- Machine Learning prediction (Random Forest)
- Real-time temperature (API)

**Weighted Formula:**


---

## Technologies Used
- Python
- Streamlit
- Pandas, NumPy
- Matplotlib, Seaborn
- Plotly
- Scikit-learn
- SciPy
- Statsmodels

---

## How to Run
-run directly on colab (uses streamlit & ngrok)

<img width="1244" height="748" alt="image" src="https://github.com/user-attachments/assets/4d35bf8e-b336-46e5-b834-6e6720aaea40" />
<img width="1179" height="541" alt="image" src="https://github.com/user-attachments/assets/eb7b4de8-0093-4e1f-a4a7-d5ec0e0ca5a5" />
<img width="1218" height="368" alt="image" src="https://github.com/user-attachments/assets/5f4dc729-b317-4d32-9a03-0bb51ce22c7f" />
<img width="1125" height="764" alt="image" src="https://github.com/user-attachments/assets/5d68809f-1655-43c8-90b3-5561efce9618" />
<img width="864" height="795" alt="image" src="https://github.com/user-attachments/assets/159fbeae-04bd-461f-800b-5a70894c7e1c" />
<img width="1154" height="340" alt="image" src="https://github.com/user-attachments/assets/a617a843-351f-4adb-b7e0-931b306dee4a" />
<img width="1205" height="559" alt="image" src="https://github.com/user-attachments/assets/83e82143-f656-4a48-bb99-900beb497a28" />
