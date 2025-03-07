import numpy as np
import utils
import streamlit as st

st.title('Autoregressive process simulation and forecasting')

# Parameters
st.sidebar.markdown('# Parameters')
n_steps = st.sidebar.number_input('Number of time steps to simulate', min_value=100, max_value=10000, value=1000, step=100)
p_true = st.sidebar.slider('Order of the AR process ($p$)', min_value=1, max_value=20, value=5, step=1)

# Slider for modulus of the roots
Lr, Ur = st.sidebar.slider(
    'Modulus of the roots range',
    min_value=0.0, max_value=1.0, value=(0.2, 0.8), step=0.1
)

# Slider for angle of the roots
Ltheta, Utheta = st.sidebar.slider(
    'Angle of the roots range',
    min_value=0, max_value=180, value=(0, 40), step=1
)
seed = st.sidebar.number_input('Seed for random number generation', min_value=0, max_value=1000, value=42, step=1)


# Forecasting parameters
st.sidebar.markdown('# Forecasting parameters')
p_forecast = st.sidebar.slider('Order of the AR process for forecasting', min_value=1, max_value=20, value=5, step=1)
forecast_horizon = st.sidebar.slider('Forecast horizon', min_value=1, max_value=10, value=5, step=1)
submit_button = st.sidebar.button('Generate and forecast the AR process')
# forecast_stride = st.sidebar.number_input('Stride for forecasting', min_value=1, max_value=100, value=1, step=1)

import streamlit as st

st.markdown("""
## **Synthesizing Time Series Using AR**

Autoregressive (AR) models are widely used for synthesizing and analyzing time series data. 
An AR model assumes that the current value of a time series depends linearly on its past values, with some added random noise. 
This makes it a powerful tool for simulating stationary time series.

---
### **1. Conceptual Overview**
An AR(p) process is defined by its order $ p $, which specifies the number of lagged terms (past values) used to predict the current value. To synthesize a time series using an AR(p) model:
- We define the AR coefficients ($ \phi_1, \phi_2, ..., \phi_p $).
- We generate random noise ($ \epsilon_t $) from a normal distribution.
- Using the AR equation, we iteratively compute each value in the time series based on the previous $ p $ values.

---
### **2. Mathematical Formulation**
The AR(p) process is mathematically expressed as:

$$
X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + ... + \phi_p X_{t-p} + \epsilon_t
$$

Where:
- $ X_t $: The value of the time series at time $ t $.
- $ \phi_1, \phi_2, ..., \phi_p $: The AR coefficients that determine the influence of past values.
- $ \epsilon_t $: A random noise term drawn from a normal distribution, typically $ \epsilon_t \sim \mathcal{N}(0, \sigma^2) $.

For simulation purposes, the initial $ p $ values ($ X_0, X_1, ..., X_{p-1} $) are often set as random values or zeros.

---
### **3. Stationarity Condition**

To ensure that an autoregressive (AR) process is stationary, we analyze the roots of the characteristic polynomial associated with the AR equation. A stationary process has statistical properties (e.g., mean, variance, autocovariance) that do not change over time. The stationarity condition imposes constraints on the AR coefficients ($ \phi_1, \phi_2, ..., \phi_p $).

---
#### **3.1. Why Stationarity Matters**
A stationary time series is crucial for many time series analyses because:
- Its statistical properties remain constant over time, making it easier to model and forecast.
- Non-stationary processes can exhibit trends or seasonality, which complicate analysis unless explicitly accounted for.

For an AR(p) process, stationarity is determined by the roots of the **characteristic polynomial**, as explained below.

---

#### **3.2. Derivation of the Characteristic Polynomial**

The AR(p) process is defined as:

$$
X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + ... + \phi_p X_{t-p} + \epsilon_t
$$

Rearranging terms to isolate $ X_t $, we get:

$$
X_t - \phi_1 X_{t-1} - \phi_2 X_{t-2} - ... - \phi_p X_{t-p} = \epsilon_t
$$

To analyze the behavior of this process without the noise term ($ \epsilon_t $), we consider the **homogeneous equation** (i.e., setting $ \epsilon_t = 0 $):

$$
X_t - \phi_1 X_{t-1} - \phi_2 X_{t-2} - ... - \phi_p X_{t-p} = 0
$$

This equation describes how the time series evolves purely based on its past values. To solve it, we assume a solution of the form:

$$
X_t = z^t
$$

Substituting $ X_t = z^t $ into the homogeneous equation gives:

$$
z^t - \phi_1 z^{t-1} - \phi_2 z^{t-2} - ... - \phi_p z^{t-p} = 0
$$

Factoring out $ z^{t-p} $ (since $ z \neq 0 $), we obtain:

$$
z^p - \phi_1 z^{p-1} - \phi_2 z^{p-2} - ... - \phi_p = 0
$$

This is the **characteristic polynomial** of the AR(p) process:

$$
P(z) = z^p - \phi_1 z^{p-1} - \phi_2 z^{p-2} - ... - \phi_p
$$

---

#### **3.3. Roots of the Characteristic Polynomial**

The roots of the characteristic polynomial ($ z_1, z_2, ..., z_p $) are solutions to the equation:

$$
P(z) = 0
$$

These roots determine the behavior of the AR process:
- If all roots satisfy $ |z| < 1 $ (i.e., they lie inside the unit circle in the complex plane), the process is **stationary**.
- If any root satisfies $ |z| \geq 1 $, the process is **non-stationary**.

---

#### **3.4. Intuition Behind the Stationarity Condition**

The stationarity condition ensures that the influence of past values diminishes over time. Here’s why:

1. **Behavior of Roots**:
   - Each root $ z_i $ corresponds to a component of the time series of the form $ C z_i^t $, where $ C $ is a constant.
   - If $ |z_i| < 1 $, then $ z_i^t $ decays exponentially as $ t \to \infty $, ensuring that the contribution of past values fades over time.
   - If $ |z_i| \geq 1 $, the contribution does not decay (or grows), leading to non-stationarity.

2. **Impact on Variance**:
   - For a stationary process, the variance of $ X_t $ remains constant over time.
   - If any root lies outside or on the unit circle, the variance may grow indefinitely, violating stationarity.

---

### **4. Simulation Steps**
1. **Define Parameters**:
   - Choose the order $ p $ of the AR process.
   - Specify the AR coefficients ($ \phi_1, \phi_2, ..., \phi_p $).
   - Set the standard deviation $ \sigma $ of the noise term $ \epsilon_t $.

2. **Initialize Values**:
   - Generate the first $ p $ values of the time series ($ X_0, X_1, ..., X_{p-1} $).

3. **Iterate Using the AR Equation**:
   - For each time step $ t $, compute $ X_t $ using:
     $$
     X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + ... + \phi_p X_{t-p} + \epsilon_t
     $$

4. **Add Noise**:
   - Draw $ \epsilon_t $ from $ \mathcal{N}(0, \sigma^2) $ at each step.

### **5. Example**
For an AR(2) process with coefficients $ \phi_1 = 0.6 $, $ \phi_2 = -0.4 $, and noise $ \epsilon_t \sim \mathcal{N}(0, 1) $, the equation becomes:

$$
X_t = 0.6 X_{t-1} - 0.4 X_{t-2} + \epsilon_t
$$

By iterating this equation, we can generate a synthetic time series.

---

### **6. Parameters for Synthesizing Time Series**

To create a synthetic time series using an Autoregressive (AR) model, you can adjust the following parameters in the sidebar:

1. **Number of Time Steps to Simulate**:

2. **Order of the AR Process, (p)**:

3. **Modulus of the Roots Range**:
    - These sliders control the range for the modulus of the roots of the characteristic polynomial.
    - The modulus values determine the stability and stationarity of the AR process.
    - You can set the lower and upper bounds of the modulus range from 0.0 to 1.0.
    - The roots are drawn randomly from the specified range.

4. **Angle of the Roots Range**:
    - These sliders control the range for the angle of the roots of the characteristic polynomial.
    - The angle values influence the periodicity and oscillatory behavior of the AR process.
    - You can set the lower and upper bounds of the angle range from 0 to 180 degrees.
    - The roots are drawn randomly from the specified range.

5. **Seed for Random Number Generation**:
    - This parameter sets the seed for the random number generator to ensure reproducibility.

6. **Forecasting Parameters**:
    - **Order of the AR Process for Forecasting**:
        - This specifies the number of lagged terms used for forecasting the time series.
        - It doesn't have to be the same as the order of the AR process used for simulation. In forecasting stage, the actual order of the AR process is unknown.
    - **Forecast Horizon**:
        - This determines the number of steps ahead to forecast.
        - You can set the forecast horizon from 1 to 10 steps.
        - When the horizon is larger, the uncertainty in the forecast increases.


By adjusting these parameters in the sidebar, you can control the characteristics of the synthetic time series generated by the AR model.
""")

if submit_button:
    # convert the angle to radians
    Ltheta = np.radians(Ltheta)
    Utheta = np.radians(Utheta)

    # Generate stationary phis and roots
    phis, roots = utils.generate_stationary_phis(p_true, Lr, Ur, Ltheta, Utheta, seed=seed)

    # Simulate the AR(p) process
    X = utils.simulate_ar_process(phis, n_steps, seed=seed)

    # Estimate the AR coefficients using least squares
    estimated_phis = utils.estimate_ar_coefficients(X, p_forecast)

    # forecast using the AR process for a given number of steps using original and estimated phis
    forecast_stride = forecast_horizon
    forecasts = [utils.forecast_ar_with_ci(
            estimated_phis, 
            initial_values = X[i:i+p_forecast], 
            n_forecast = forecast_horizon,
            sigma_epsilon = 1,
            confidence_level = 0.95,
            ) for i in range(0, n_steps-forecast_horizon, forecast_stride)]
    X_forecast = [f[0] for f in forecasts]
    LB = [f[1] for f in forecasts]
    HB = [f[2] for f in forecasts]
    time_vector_forecast = [np.arange(i + p_true, i + p_true + forecast_horizon) for i in range(0, n_steps-forecast_horizon, forecast_stride)]

    fig_roots = utils.plot_roots(roots, Lr, Ur, Ltheta, Utheta)
    fig_AR = utils.plot_ar_process(X, X_forecast, time_vector_forecast, LB, HB)
    fig_comp = utils.plot_forecast_vs_true(X, X_forecast, time_vector_forecast)
    # fig_phi = plot_ar_coefficient_comparison(p, phis, estimated_phis)
    fig_hist = utils.plot_forecast_vs_true_historam(X, X_forecast, time_vector_forecast)

    st.plotly_chart(fig_roots)
    st.plotly_chart(fig_AR)
    st.plotly_chart(fig_comp)
    st.plotly_chart(fig_hist)