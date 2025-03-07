import numpy as np
import plotly.graph_objects as go
import matplotlib.cm as cm
import plotly.io as pio
from scipy.special import erfinv
import matplotlib.pyplot as plt
from scipy import integrate

pio.templates.default = "plotly"
colors = cm.Set1.colors        # options: tab10, tab20, tab20b, tab20c, viridis, inferno, magma, plasma, cividis
colors = [f'rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, 1)' for r, g, b in colors]

# Function to generate stationary AR(p) coefficients
def generate_stationary_phis(p, Lr=0.01, Ur=0.99, Ltheta=0, Utheta = np.pi, seed=None):
    """
    Generate stationary AR(p) coefficients by sampling roots outside the unit circle.
    
    Parameters:
        p (int): Order of the AR process.
        
    Returns:
        phis (ndarray): Stationary AR coefficients [phi1, phi2, ..., phip].
        roots (ndarray): Roots of the characteristic equation.
    """
    if seed is not None: np.random.seed(seed)
    # Step 1: Sample roots inside the unit circle
    roots = []
    for _ in range(p // 2):  # Generate conjugate pairs for complex roots
        r = np.random.uniform(Lr, Ur)  # Modulus < 1
        theta = np.random.uniform(Ltheta, Utheta)  # Random angle
        root = r * np.exp(1j * theta)  # Complex root
        roots.append(root)
        roots.append(np.conj(root))  # Add conjugate pair
    
    if p % 2 == 1:  # If p is odd, add one real root
        if Utheta < np.pi/2:
            p_pos = 1
        elif Ltheta > np.pi/2:
            p_pos = 0
        else:
            p_pos = (np.pi/2 - Ltheta) / (Utheta - Ltheta)

        sign = np.random.choice([1, -1], p=[p_pos, 1-p_pos])
        min_r = min(sign*Lr, sign*Ur)  # Minimum value
        max_r = max(sign*Lr, sign*Ur)  # Maximum value
        r = np.random.uniform(min_r, max_r) 
        roots.append(r)
    
    roots = np.array(roots)
    
    # Step 2: Compute the polynomial coefficients from the roots
    poly_coeffs = np.poly(roots)  # np.poly computes the coefficients from roots
    phis = -poly_coeffs[1:]  # Extract phi coefficients (ignore the leading 1)
    
    return phis, roots

# Function to simulate an AR(p) process
def simulate_ar_process(phis, n_steps, initial_values=None, seed=None):
    if seed is not None: np.random.seed(seed)

    p = len(phis)
    if initial_values is None:
        initial_values = np.random.normal(size=p)  # Random initial values
    
    # Initialize the time series
    X = np.zeros(n_steps)
    X[:p] = initial_values  # Set initial values
    
    # Simulate the AR(p) process
    for t in range(p, n_steps):
        X[t] = np.dot(phis, X[t-p:t][::-1])  + np.random.normal()  # Add noise term

    return X

# Function to estimate AR coefficients using least squares
def estimate_ar_coefficients(time_series, p):
    """
    Estimate the coefficients of an AR(p) model using least squares.
    
    Parameters:
        time_series (ndarray): The input time series.
        p (int): Order of the AR model.
        
    Returns:
        phis (ndarray): Estimated AR coefficients [phi1, phi2, ..., phip].
    """
    n = len(time_series)
    if n <= p:
        raise ValueError("Time series length must be greater than the order p.")
    
    # Step 1: Construct the design matrix (lagged values)
    X = np.array([time_series[i:i+p][::-1] for i in range(n - p)])
    
    # Step 2: Construct the target vector (next values)
    y = time_series[p:]
    
    # Step 3: Solve the least squares problem
    phis = np.linalg.lstsq(X, y, rcond=None)[0]
    
    return phis

# Compare the real phis with the estimated ones in a bar graph
def plot_ar_coefficient_comparison(p, phis, estimated_phis):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=np.arange(1, p+1), 
        y=phis, 
        width=0.4, 
        name='Real Phis',
    ))
    fig.add_trace(go.Bar(
        x=np.arange(1, p+1), 
        y=estimated_phis, 
        width=0.4, 
        name='Estimated Phis',
    ))
    fig.update_layout(
        title='Comparison of Real and Estimated AR Coefficients', 
        xaxis_title='Coefficient Index', 
        yaxis_title='Value')
    return fig

# Plot the AR process
def plot_ar_process(X, X_forecast, time_vector_forecast, LB=None, HB=None):
    n_steps = len(X)
    if n_steps > 200:
        X = X[:200]
        n_steps = 200

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y = X,
        mode='lines', 
        name='AR Process',
        line=dict(color=colors[0], width=2),
        ))
    fig.add_trace(go.Scatter(
        y=[0]*(n_steps), 
        mode='lines', 
        name='Zero Line', 
        line=dict(color='grey', dash='dash'), 
        showlegend=False,
        ))

    i, ci = 0, 0
    for i in range(len(X_forecast)):
        if time_vector_forecast[i][-1] >= n_steps: break
        if LB is not None:
            fig.add_trace(go.Scatter(
                x=time_vector_forecast[i],
                y=LB[i],
                mode='lines', 
                name='Lower Bound', 
                line=dict(color=colors[1], width=1),
                showlegend=False,
            ))
        fig.add_trace(go.Scatter(
            x=time_vector_forecast[i],
            y=X_forecast[i], 
            mode='lines', 
            name='Forecast', 
            line=dict(color=colors[1], width=1),
            showlegend=False,
            fill='tonexty' if LB is not None else 'none',
            ))
        if HB is not None:
            fig.add_trace(go.Scatter(
                x=time_vector_forecast[i],
                y=HB[i],
                mode='lines', 
                name='Upper Bound', 
                line=dict(color=colors[1], width=1),
                fill='tonexty',
                showlegend=False,
            ))
        i += 1
        # ci += 1
        # if ci == len(colors) - 1: ci = 0

    fig.update_layout(title='Simulated AR(p) Process', xaxis_title='Time', yaxis_title='Value')
    return fig

# plot the roots and the unit circle
def plot_roots(roots, Lr, Ur, Ltheta, Utheta):
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=np.abs(roots), 
        theta=np.angle(roots, deg=True), 
        mode='markers', 
        marker=dict(size=5, color='black'), 
        name='Roots',
    ))
    
    # Plot the range from Lr to Ur and from Ltheta to Utheta
    theta_range = np.linspace(np.degrees(Ltheta), np.degrees(Utheta), 100)
    fig.add_trace(go.Scatterpolar(
        r=[Lr]*100, 
        theta=theta_range, 
        mode='lines', 
        line=dict(color='red', dash='dash'), 
        name='Lower Radius',
        showlegend=False,
    ))
    fig.add_trace(go.Scatterpolar(
        r=[Ur]*100, 
        theta=theta_range, 
        mode='lines', 
        line=dict(color='red', dash='dash'), 
        name='Upper Radius',
        showlegend=False,
    ))
    fig.add_trace(go.Scatterpolar(
        r=np.linspace(Lr, Ur, 100), 
        theta=[np.degrees(Ltheta)]*100, 
        mode='lines', 
        line=dict(color='red', dash='dash'), 
        name='Lower Theta',
        showlegend=False,
    ))
    fig.add_trace(go.Scatterpolar(
        r=np.linspace(Lr, Ur, 100), 
        theta=[np.degrees(Utheta)]*100, 
        mode='lines', 
        line=dict(color='red', dash='dash'), 
        name='Upper Theta',
        showlegend=False,
    ))
    
    # lower side of the figure
    fig.add_trace(go.Scatterpolar(
        r=[Lr]*100, 
        theta=-theta_range, 
        mode='lines', 
        line=dict(color='red', dash='dash'), 
        name='Lower Radius',
        showlegend=False,
    ))
    fig.add_trace(go.Scatterpolar(
        r=[Ur]*100, 
        theta=-theta_range, 
        mode='lines', 
        line=dict(color='red', dash='dash'), 
        name='Upper Radius',
        showlegend=False,
    ))
    fig.add_trace(go.Scatterpolar(
        r=np.linspace(Lr, Ur, 100), 
        theta=[np.degrees(-Ltheta)]*100, 
        mode='lines', 
        line=dict(color='red', dash='dash'), 
        name='Lower Theta',
        showlegend=False,
    ))
    fig.add_trace(go.Scatterpolar(
        r=np.linspace(Lr, Ur, 100), 
        theta=[np.degrees(-Utheta)]*100, 
        mode='lines', 
        line=dict(color='red', dash='dash'), 
        name='Upper Theta',
        showlegend=False,
    ))

    # update layout
    fig.update_layout(title='Roots of the Characteristic Equation', polar=dict(radialaxis=dict(visible=True)))

    return fig

# forecast future values of an AR(p) process with confidence intervals
def forecast_ar_with_ci(phis, initial_values, n_forecast, sigma_epsilon=1, confidence_level=0.95):
    """
    Forecast future values of an AR(p) process with confidence intervals.
    
    Parameters:
        phis (ndarray): AR coefficients [phi1, phi2, ..., phip].
        initial_values (ndarray): Initial values for forecasting.
        n_forecast (int): Number of steps to forecast.
        sigma_epsilon (float): Standard deviation of the noise term.
        confidence_level (float): Confidence level for intervals (e.g., 0.95 for 95% CI).
        
    Returns:
        forecast (ndarray): Forecasted time series.
        lower_ci (ndarray): Lower bounds of confidence intervals.
        upper_ci (ndarray): Upper bounds of confidence intervals.
    """
    p = len(phis)
    total_length = p + n_forecast
    forecast = np.zeros(total_length)
    forecast_var = np.zeros(total_length)
    lower_ci = np.zeros(total_length)
    upper_ci = np.zeros(total_length)
    
    # Initialize with initial values
    forecast[:p] = initial_values[-p:]
    forecast_var[:p] = 0  # No uncertainty for initial values
    
    # Critical value for confidence interval
    z = -np.sqrt(2) * erfinv(1 - confidence_level)  # Inverse CDF for normal distribution
    
    # Forecast future values and compute variance
    for t in range(p, total_length):
        forecast[t] = np.dot(phis, forecast[t-p:t][::-1])
        forecast_var[t] = sigma_epsilon**2 * (1 + np.sum(phis**2 * forecast_var[t-p:t][::-1]))
    
    # Compute confidence intervals
    lower_ci = forecast - z * np.sqrt(forecast_var)
    upper_ci = forecast + z * np.sqrt(forecast_var)
    
    return forecast[p:], lower_ci[p:], upper_ci[p:]

# Analytical formula to estimate the zero-crossing frequency of an AR(p) process
def estimate_zcf_from_roots(roots, sigma_epsilon=1.0):
    """
    Estimates the zero-crossing frequency of an AR(p) process from the roots
    of its characteristic polynomial using spectral analysis.
    
    Parameters:
    -----------
    roots : array_like
        Complex roots of the characteristic polynomial
    sigma_epsilon : float, optional
        Standard deviation of the innovation process (default=1.0)
        
    Returns:
    --------
    zcf : float
        Estimated zero-crossing frequency (crossings per sample)
    """
    # Convert roots to AR coefficients (polynomial coefficients)
    poly_coeffs = np.poly(roots)  # Gives [1, -phi_1, -phi_2, ..., -phi_p]
    ar_coeffs = -poly_coeffs[1:]  # Extract phi coefficients (ignore the leading 1)
    
    # Function to calculate spectral density at frequency omega
    def spectral_density(omega):
        p = len(ar_coeffs)
        # Calculate the denominator |1 - phi_1*e^(-iω) - ... - phi_p*e^(-ipω)|^2
        z = np.exp(-1j * omega)
        ar_term = 1.0
        for k in range(p):
            ar_term -= ar_coeffs[k] * z**(k+1)
        return sigma_epsilon**2 / np.abs(ar_term)**2
    
    # Function to calculate the integrand for spectral moments
    def integrand_lambda0(omega):
        return spectral_density(omega)
    
    def integrand_lambda2(omega):
        return omega**2 * spectral_density(omega)
    
    # Compute spectral moments λ₀ and λ₂
    lambda0, _ = integrate.quad(integrand_lambda0, 0, np.pi)
    lambda2, _ = integrate.quad(integrand_lambda2, 0, np.pi)
    
    # Account for the full spectrum (-π to π) by doubling
    lambda0 *= 2
    lambda2 *= 2
    
    # Apply Rice's formula to estimate zero-crossing rate
    # E[ZCR] = (1/π) * √(λ₂/λ₀)
    zcf = (1/np.pi) * np.sqrt(lambda2/lambda0)
    
    return zcf

def zcf_from_roots_wrapper(n_mc, num_bins, p):

    # Generate bin edges for x and y (0.0, 0.1, ..., 1.0)
    r_edges = np.linspace(0, 1, num_bins + 1)
    t_edges = np.linspace(0, np.pi, num_bins + 1)
    zcf = np.zeros((num_bins, num_bins))
    # Initialize a list to collect all points

    # Iterate over each x and y bin
    for i in range(num_bins):
        r_start, r_end = r_edges[i], r_edges[i+1]
        for j in range(num_bins):
            t_start, t_end = t_edges[j], t_edges[j+1]
            
            all_points = []
            # MC approach
            for k in range(n_mc):
                # Generate stationary phis
                phis, roots = generate_stationary_phis(p, Lr=r_start, Ur=r_end, Ltheta=t_start, Utheta=t_end, seed=None)
                
                # Simulate the AR(p) process
                # X = simulate_ar_process(phis, n_steps)

                # Count the number of zero crossings
                # zero_crossings = np.sum(np.abs(np.diff(np.sign(X))) / 2)

                # Calculate the zero crossing frequency
                # zero_crossing_frequency = zero_crossings / (n_steps - 1)
                zero_crossing_frequency = estimate_zcf_from_roots(roots, sigma_epsilon=1.0)

                # Append the point to the list
                all_points.append([zero_crossing_frequency, np.mean(phis.imag)])
            
            zcf[i, j] = np.mean(all_points, axis=0)[0]


    # Create a meshgrid for polar coordinates
    r_centres = (r_edges[:-1] + r_edges[1:]) / 2
    t_centres = (t_edges[:-1] + t_edges[1:]) / 2
    R, Theta = np.meshgrid(r_centres, t_centres)

    # Create a contour plot in polar coordinates
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    contour = ax.contourf(Theta, R, zcf.T, cmap='viridis')

    # Add a colorbar
    cbar = plt.colorbar(contour)
    cbar.set_label('Zero Crossing Frequency')

    # Set plot title
    ax.set_title('Zero Crossing Frequency in Polar Coordinates')

    plt.show()

def plot_forecast_vs_true(X, X_forecast, time_vector_forecast):
    fig = go.Figure()
    for i in range(len(X_forecast[:-1])):
        fig.add_trace(go.Scatter(
            x = X[time_vector_forecast[i]],
            y = X_forecast[i],
            mode='markers',
            marker=dict(size=5, color='rgba(255, 255, 255, 0.4)'),
            showlegend=False,
        ))  
    fig.add_trace(go.Scatter(
        x=[min(X), max(X)],
        y=[min(X), max(X)],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='y=x Line'
    ))
    fig.add_hline(y=0, line_color='grey', line_width=1, name='y=0 Line')
    fig.add_vline(x=0, line_color='grey', line_width=1, name='x=0 Line')
    fig.update_layout(title='Forecast vs True Values', xaxis_title='True Value', yaxis_title='Forecast Value')
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig

def plot_forecast_vs_true_historam(X, X_forecast, time_vector_forecast):
    X_true = []
    X_fore = []
    for i in range(len(X_forecast[:-1])):
        X_true.extend(X[time_vector_forecast[i]])
        X_fore.extend(X_forecast[i])
    diff = np.array(X_fore) - np.array(X_true)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=diff, nbinsx=50))
    fig.add_vline(x=0, line_color='grey', line_width=1, name='x=0 Line')
    fig.update_layout(title='Histogram of Forecast Errors', xaxis_title='Forecast Error', yaxis_title='Frequency')
    return fig
