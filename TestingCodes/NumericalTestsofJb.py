import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# Define the function J(θ) using corrected integration
def J(theta):
    eps = 1e-6  # small offset to avoid singularities
    if theta >= 0:
        f = lambda y: y*y*np.log(1 - np.exp(-np.sqrt(y*y + theta)))
        return integrate.quad(f, 0, np.inf, limit=100)[0]
    else:
        f1 = lambda y: y*y*np.log(1 - np.exp(-np.sqrt(y*y + theta)))
        f2 = lambda y: y*y*np.log(2 * abs(np.sin(np.sqrt(abs(theta) - y*y)/2)))
        a = np.sqrt(abs(theta))
        return (
            integrate.quad(f1, a + eps, np.inf, limit=100)[0] +
            integrate.quad(f2, 0, a - eps, limit=100)[0]
        )

# Generate θ values and compute J(θ)
theta_values = np.linspace(-5, 5, 200)
J_values = np.array([J(theta) for theta in theta_values])

# Plot the result
plt.figure(figsize=(10, 6))
plt.plot(theta_values, J_values, label='J(θ) (Python)', color='blue')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.title("Plot of J(θ) over [-5, 5]")
plt.xlabel("θ")
plt.ylabel("J(θ)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Numerical derivative using central difference
def dJ_dtheta(theta, h=1e-5):
    return (J(theta + h) - J(theta - h)) / (2 * h)


# Compute derivative values over the same θ range
dJ_values = np.array([dJ_dtheta(theta) for theta in theta_values])

# Plot the derivative
plt.figure(figsize=(10, 6))
plt.plot(theta_values, dJ_values, label="dJ/dθ (numerical)", color='green')
plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.title("Numerical Derivative of J(θ) over [-5, 5]")
plt.xlabel("θ")
plt.ylabel("dJ/dθ")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
