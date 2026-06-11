
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==============================================================================
# TASK 1:  df/dn = n * sin(n),   f(0) = 0.5
# ==============================================================================
print("=" * 60)
print("TASK 1: df/dn = n·sin(n),  f(0) = 0.5")
print("=" * 60)

# Exact antiderivative:  ∫ n·sin(n) dn = -n·cos(n) + sin(n) + C
# Boundary condition f(0) = 0.5  →  C = 0.5
def f_exact_task1(n):
    return -n * np.cos(n) + np.sin(n) + 0.5

def rhs_task1(n, f):
    """Right-hand side of the ODE: df/dn = n·sin(n)."""
    return n * np.sin(n)

# --- Parameters ---
h1 = 0.1          # coarse step
h2 = 0.01         # fine step
n_start, n_end = 0, 10

# --- Euler method (h = 0.1) ---
n1 = np.arange(n_start, n_end + h1, h1)
F1 = np.zeros_like(n1)
F1[0] = 0.5
for i in range(len(n1) - 1):
    F1[i + 1] = F1[i] + h1 * rhs_task1(n1[i], F1[i])

# --- Euler method (h = 0.01) ---
n2 = np.arange(n_start, n_end + h2, h2)
F2 = np.zeros_like(n2)
F2[0] = 0.5
for i in range(len(n2) - 1):
    F2[i + 1] = F2[i] + h2 * rhs_task1(n2[i], F2[i])

# --- Built-in RK45 solver ---
sol1 = solve_ivp(rhs_task1, [n_start, n_end], [0.5], t_eval=n2, method='RK45')

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(n1, F1, 'b--', label=f'Euler (h={h1})')
plt.plot(n2, F2, 'm:', label=f'Euler (h={h2})')
plt.plot(n2, f_exact_task1(n2), 'g-', label='Exact solution')
plt.plot(sol1.t, sol1.y[0], 'r-.', label='RK45 (scipy)')
plt.xlabel('n')
plt.ylabel('f(n)')
plt.title('Task 1: df/dn = n·sin(n),  f(0) = 0.5')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()


# ==============================================================================
# TASK 2:  df/dt = exp(-x) * tan(y),   f(0,0) = 1
# ==============================================================================
print("\n" + "=" * 60)
print("TASK 2: df/dt = exp(-x)·tan(y),  f(0,0) = 1")
print("=" * 60)

# Following the lab-sheet 2D Euler approach.
# WARNING: tan(y) has singularities at ±π/2. We restrict y to a safe interval.
h = 0.05
x = np.arange(-2, 2 + h, h)
y = np.arange(-1.4, 1.4 + h, h)

f_ode = lambda x_val, y_val: np.exp(-x_val) * np.tan(y_val)

F = np.zeros((len(x), len(y)))
F[0, 0] = 1.0  # initial condition f(0,0)=1

# First column: integrate along x at fixed y = y[0]
for i in range(len(x) - 1):
    F[i + 1, 0] = F[i, 0] + h * f_ode(x[i], y[0])

# First row: integrate along y at fixed x = x[0]
for j in range(len(y) - 1):
    F[0, j + 1] = F[0, j] + h * f_ode(x[0], y[j])

# Remaining grid points (2D Euler)
for i in range(len(x) - 1):
    for j in range(len(y) - 1):
        F[i + 1, j + 1] = F[i, j] + h * f_ode(x[i], y[j])

# --- 3D surface plot ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, F.T, cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
ax.set_title('Task 2: Approximate Solution for df/dt = exp(-x)·tan(y)')
plt.tight_layout()
plt.show()


# ==============================================================================
# TASK 3: Two-compartment pharmacokinetic model
# ==============================================================================
print("\n" + "=" * 60)
print("TASK 3: PK Model (ka=0.3, ke=0.2, A0=100 mg, B0=0 mg)")
print("=" * 60)

# Parameters
ka = 0.3   # absorption rate constant  [1/min]  (unit simplified for clarity)
ke = 0.2   # elimination rate constant [1/min]
A0 = 100.0 # initial amount in bottle A [mg]
B0 = 0.0   # initial amount in bottle B [mg]

def pk_model(t, state):
    """
    Two-compartment PK model.
    state = [A, B]
    dA/dt = -ka * A
    dB/dt =  ka * A - ke * B
    """
    A, B = state
    dA_dt = -ka * A
    dB_dt = ka * A - ke * B
    return [dA_dt, dB_dt]

# Time grid
t_span = [0, 60]
h = 0.01
t_eval = np.arange(t_span[0], t_span[1] + h, h)

# --- Explicit Euler ---
F_A = np.zeros_like(t_eval)
F_B = np.zeros_like(t_eval)
F_A[0] = A0
F_B[0] = B0
for i in range(len(t_eval) - 1):
    dA, dB = pk_model(t_eval[i], [F_A[i], F_B[i]])
    F_A[i + 1] = F_A[i] + h * dA
    F_B[i + 1] = F_B[i] + h * dB

# --- Built-in RK45 solver ---
sol3 = solve_ivp(pk_model, t_span, [A0, B0], t_eval=t_eval, method='RK45')
A_rk = sol3.y[0]
B_rk = sol3.y[1]

# --- Analytical solution (valid when ka != ke) ---
A_ana = A0 * np.exp(-ka * t_eval)
B_ana = (ka * A0) / (ke - ka) * (np.exp(-ka * t_eval) - np.exp(-ke * t_eval))

# --- Find when B drops below 1 mg AFTER reaching its peak ---
idx_max = np.argmax(B_rk)
t_max = t_eval[idx_max]
B_max = B_rk[idx_max]

idx_below_after_peak = np.where((B_rk < 1.0) & (t_eval > t_max))[0]
if len(idx_below_after_peak) > 0:
    t_below = t_eval[idx_below_after_peak[0]]
    print(f"\n>>> B peaks at t = {t_max:.2f} min with value = {B_max:.4f} mg")
    print(f">>> B drops below 1 mg (after peak) at t = {t_below:.2f} minutes")
else:
    t_below = None
    print(f"\n>>> B peaks at t = {t_max:.2f} min with value = {B_max:.4f} mg")
    print(">>> B never drops below 1 mg after the peak in the simulated interval.")

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Time series
ax1.plot(t_eval, F_A, 'b--', label='A(t) – Euler', alpha=0.7)
ax1.plot(t_eval, F_B, 'r--', label='B(t) – Euler', alpha=0.7)
ax1.plot(t_eval, A_rk, 'b-', linewidth=2, label='A(t) – RK45')
ax1.plot(t_eval, B_rk, 'r-', linewidth=2, label='B(t) – RK45')
ax1.plot(t_eval, A_ana, 'g:', linewidth=1.5, label='A(t) – Analytical')
ax1.plot(t_eval, B_ana, 'k:', linewidth=1.5, label='B(t) – Analytical')
ax1.axhline(1.0, color='purple', linestyle='-', alpha=0.5, label='Threshold = 1 mg')
if t_below:
    ax1.axvline(t_below, color='orange', linestyle='--', alpha=0.7)
    ax1.scatter([t_below], [1.0], color='orange', zorder=5, s=80)
    ax1.scatter([t_max], [B_max], color='green', zorder=5, s=80, marker='^')
ax1.set_xlabel('Time [min]')
ax1.set_ylabel('Amount [mg]')
ax1.set_title('Concentration vs Time')
ax1.legend(loc='upper right')
ax1.grid(True)

# Right: Phase portrait (A vs B)
ax2.plot(F_A, F_B, 'm--', label='Euler', alpha=0.6)
ax2.plot(A_rk, B_rk, 'c-', linewidth=2, label='RK45')
ax2.set_xlabel('A [mg]')
ax2.set_ylabel('B [mg]')
ax2.set_title('Phase Portrait: A vs B')
ax2.legend(loc='best')
ax2.grid(True)

title_str = f'Task 3: PK Model | Peak B={B_max:.2f} mg at t={t_max:.1f} min'
if t_below:
    title_str += f' | B < 1 mg at t ≈ {t_below:.2f} min'
fig.suptitle(title_str, fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
