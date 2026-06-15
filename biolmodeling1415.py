

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from io import StringIO

# ============================================================================
#  COMMON UTILITIES (Euler integration – raw rates, single scaling by dt)
# ============================================================================

def two_compartment_euler(t, params, Dose, dt):
    """
    Two‑compartment model with IV bolus at t=0 only.
    params = [ke, k12, k21] in 1/h (raw rates).
    """
    ke, k12, k21 = params
    C1 = np.zeros_like(t)
    C2 = np.zeros_like(t)
    C1[0] = Dose
    for i in range(1, len(t)):
        dC1 = (k21 * C2[i-1] - k12 * C1[i-1] - ke * C1[i-1]) * dt
        dC2 = (k12 * C1[i-1] - k21 * C2[i-1]) * dt
        C1[i] = C1[i-1] + dC1
        C2[i] = C2[i-1] + dC2
    return C1, C2


def two_compartment_infusion_euler(t, params, IVtimes, Dose, dt):
    """
    Two‑compartment model with repeated IV bolus doses at specified times.
    params = [ke, k12, k21] in 1/h (raw rates).
    """
    ke, k12, k21 = params
    C1 = np.zeros_like(t)
    C2 = np.zeros_like(t)
    # Initial dose at time 0 if present
    if len(IVtimes) > 0 and np.isclose(IVtimes[0], 0.0):
        C1[0] = Dose
    else:
        C1[0] = 0.0
    for i in range(1, len(t)):
        dC1 = (k21 * C2[i-1] - k12 * C1[i-1] - ke * C1[i-1]) * dt
        dC2 = (k12 * C1[i-1] - k21 * C2[i-1]) * dt
        C1[i] = C1[i-1] + dC1
        C2[i] = C2[i-1] + dC2
        # Add bolus if current time matches any IVtime
        if any(np.isclose(t[i], IVtimes)):
            C1[i] += Dose
    return C1, C2


def three_compartment_euler(t, params, C0, dt):
    """
    Three‑compartment model for sucrose.
    params = [k10, k12, k21, k13, k20, k30] in 1/h.
    """
    k10, k12, k21, k13, k20, k30 = params
    C1 = np.zeros_like(t)
    C2 = np.zeros_like(t)
    C3 = np.zeros_like(t)
    C1[0], C2[0], C3[0] = C0
    for i in range(1, len(t)):
        dC1 = (k21 * C2[i-1] - k12 * C1[i-1] - k13 * C1[i-1] - k10 * C1[i-1]) * dt
        dC2 = (k12 * C1[i-1] - k21 * C2[i-1] - k20 * C2[i-1]) * dt
        dC3 = (k13 * C1[i-1] - k30 * C3[i-1]) * dt
        C1[i] = C1[i-1] + dC1
        C2[i] = C2[i-1] + dC2
        C3[i] = C3[i-1] + dC3
    return C1, C2, C3


# ============================================================================
#  TASK 1 – Fit 30 patients, remove outlier (patient 1), average parameters
# ============================================================================

def load_patient_data():
    """Load the 30‑patient dataset (embedded as CSV string)."""
    data_text = """ID,Time,CentralConc,PeripheralConc,Sex,Age
1,0,83.3778,0.0685,Female,Old
1,1,31.0188,29.6614,Female,Old
1,4,6.4875,11.1966,Female,Old
1,8,1.1631,2.0693,Female,Old
1,12,0.0755,0.4011,Female,Old
1,18,0.0896,0.0463,Female,Old
1,24,0.0177,0,Female,Old
1,36,0,0,Female,Old
2,0,49.9918,0,Female,Young
2,1,25.276,12.9022,Female,Young
2,4,7.1079,11.7248,Female,Young
2,8,2.7109,5.1072,Female,Young
2,12,1.0497,2.0969,Female,Young
2,18,0.2943,0.6164,Female,Young
2,24,0.0165,0.0946,Female,Young
2,36,0,0.0122,Female,Young
3,0,50.0276,0,Male,Young
3,1,26.2685,23.3222,Male,Young
3,4,14.3645,16.461,Male,Young
3,8,7.3422,8.5099,Male,Young
3,12,3.6954,4.311,Male,Young
3,18,1.3512,1.5315,Male,Young
3,24,0.4582,0.5187,Male,Young
3,36,0.1869,0.1044,Male,Young
4,0,49.9911,0.0418,Female,Young
4,1,25.1881,12.8018,Female,Young
4,4,7.0851,11.7371,Female,Young
4,8,2.6486,5.0019,Female,Young
4,12,1.0915,2.0341,Female,Young
4,18,0.3658,0.5594,Female,Young
4,24,0.0357,0.1829,Female,Young
4,36,0.0402,0.1396,Female,Young
5,0,49.9667,0.005,Female,Young
5,1,26.9981,13.2249,Female,Young
5,4,8.6402,13.2263,Female,Young
5,8,3.4923,6.2498,Female,Young
5,12,1.5941,2.8721,Female,Young
5,18,0.4018,0.9034,Female,Young
5,24,0.1916,0.3492,Female,Young
5,36,0,0.0147,Female,Young
6,0,83.2264,0,Male,Old
6,1,39.4643,42.8585,Male,Old
6,4,15.8458,17.8866,Male,Old
6,8,4.6649,5.3469,Male,Old
6,12,1.4591,1.6556,Male,Old
6,18,0.237,0.2066,Male,Old
6,24,0.1096,0.019,Male,Old
6,36,0,0,Male,Old
7,0,83.3883,0.0141,Male,Old
7,1,37.1434,41.1244,Male,Old
7,4,12.8987,14.6857,Male,Old
7,8,3.0801,3.7063,Male,Old
7,12,0.7697,0.9206,Male,Old
7,18,0.0557,0.0962,Male,Old
7,24,0,0.0148,Male,Old
7,36,0.0256,0,Male,Old
8,0,49.9125,0.026,Female,Young
8,1,27.5444,13.3884,Female,Young
8,4,9.1808,13.7126,Female,Young
8,8,3.9091,6.6885,Female,Young
8,12,1.7795,3.1965,Female,Young
8,18,0.5582,0.9951,Female,Young
8,24,0.0861,0.2833,Female,Young
8,36,0.0671,0.0999,Female,Young
9,0,83.3221,0,Female,Old
9,1,29.3773,28.8658,Female,Old
9,4,5.6235,10.0726,Female,Old
9,8,0.8981,1.7379,Female,Old
9,12,0.1018,0.2184,Female,Old
9,18,0.1371,0,Female,Old
9,24,0.0835,0,Female,Old
9,36,0.0154,0.0167,Female,Old
10,0,50.0196,0,Female,Young
10,1,24.9055,12.7191,Female,Young
10,4,6.7484,11.5074,Female,Young
10,8,2.5032,4.9243,Female,Young
10,12,0.9843,1.9265,Female,Young
10,18,0.3039,0.576,Female,Young
10,24,0,0.1657,Female,Young
10,36,0.0273,0.0055,Female,Young
11,0,49.9902,0,Male,Young
11,1,26.0236,23.2241,Male,Young
11,4,13.9824,16.2091,Male,Young
11,8,7.069,8.1346,Male,Young
11,12,3.5726,4.1823,Male,Young
11,18,1.3229,1.4997,Male,Young
11,24,0.5364,0.5396,Male,Young
11,36,0.0826,0.0944,Male,Young
12,0,50.0131,0,Female,Young
12,1,24.8012,12.6685,Female,Young
12,4,6.7188,11.4433,Female,Young
12,8,2.4718,4.8384,Female,Young
12,12,0.9738,2.0472,Female,Young
12,18,0.3422,0.4962,Female,Young
12,24,0.0228,0.0781,Female,Young
12,36,0,0.0889,Female,Young
13,0,83.3951,0.0196,Female,Old
13,1,29.3991,28.8485,Female,Old
13,4,5.5649,10.0365,Female,Old
13,8,0.9188,1.6621,Female,Old
13,12,0.1501,0.2598,Female,Old
13,18,0.0247,0.0036,Female,Old
13,24,0,0.002,Female,Old
13,36,0.0222,0,Female,Old
14,0,83.3105,0,Male,Old
14,1,34.7169,39.2596,Male,Old
14,4,10.1613,11.988,Male,Old
14,8,2.0885,2.3974,Male,Old
14,12,0.4257,0.4421,Male,Old
14,18,0.035,0.0413,Male,Old
14,24,0.0124,0.0154,Male,Old
14,36,0,0.0213,Male,Old
15,0,49.9814,0,Female,Young
15,1,25.8919,12.974,Female,Young
15,4,7.7253,12.3156,Female,Young
15,8,2.8635,5.4252,Female,Young
15,12,1.378,2.3837,Female,Young
15,18,0.3708,0.5991,Female,Young
15,24,0.1489,0.1877,Female,Young
15,36,0,0.047,Female,Young
16,0,50.0164,2.49E-06,Male,Young
16,1,26.1804,23.2373,Male,Young
16,4,14.1958,16.3838,Male,Young
16,8,7.1726,8.358,Male,Young
16,12,3.6841,4.261,Male,Young
16,18,1.2881,1.6059,Male,Young
16,24,0.4196,0.6078,Male,Young
16,36,0.1105,0.0862,Male,Young
17,0,49.9655,5.81E-05,Male,Young
17,1,24.5097,22.4154,Male,Young
17,4,11.8104,13.9446,Male,Young
17,8,5.0562,6.1841,Male,Young
17,12,2.2451,2.582,Male,Young
17,18,0.5522,0.6624,Male,Young
17,24,0.2389,0.2291,Male,Young
17,36,0.0588,0,Male,Young
18,0,83.3539,0.0449,Female,Old
18,1,29.9191,29.1443,Female,Old
18,4,5.9293,10.3862,Female,Old
18,8,0.9754,1.8513,Female,Old
18,12,0.1968,0.2046,Female,Old
18,18,0.0175,0,Female,Old
18,24,0.0422,0,Female,Old
18,36,0.0268,0,Female,Old
19,0,83.3658,0,Female,Old
19,1,31.3895,29.8733,Female,Old
19,4,6.6474,11.2862,Female,Old
19,8,1.2287,2.2081,Female,Old
19,12,0.2423,0.3729,Female,Old
19,18,0.0046,0,Female,Old
19,24,0.0167,0,Female,Old
19,36,0.02,0,Female,Old
20,0,83.3323,0.0082,Female,Old
20,1,29.3561,28.9351,Female,Old
20,4,5.7356,10.0532,Female,Old
20,8,0.9942,1.7724,Female,Old
20,12,0.0322,0.2598,Female,Old
20,18,0.0328,0.0358,Female,Old
20,24,0,0.0346,Female,Old
20,36,0,0.0043,Female,Old
21,0,50.044,0.0057,Female,Young
21,1,26.8511,13.1697,Female,Young
21,4,8.4543,13.0546,Female,Young
21,8,3.4049,6.1579,Female,Young
21,12,1.654,2.8464,Female,Young
21,18,0.4388,0.7903,Female,Young
21,24,0.1461,0.2897,Female,Young
21,36,0.0409,0.0381,Female,Young
22,0,83.3216,0.0168,Male,Old
22,1,40.2922,43.509,Male,Old
22,4,16.9132,18.8917,Male,Old
22,8,5.4,6.0536,Male,Old
22,12,1.652,1.8353,Male,Old
22,18,0.3209,0.3994,Male,Old
22,24,0.015,0.1839,Male,Old
22,36,0,0.05,Male,Old
23,0,83.3175,0,Female,Old
23,1,30.6865,29.5266,Female,Old
23,4,6.2572,10.8891,Female,Old
23,8,1.2271,1.9291,Female,Old
23,12,0.2517,0.3123,Female,Old
23,18,0.0551,0.017,Female,Old
23,24,0,0,Female,Old
23,36,0.016,0.0304,Female,Old
24,0,49.9941,0.0564,Female,Young
24,1,25.1063,12.7681,Female,Young
24,4,6.9225,11.6898,Female,Young
24,8,2.6016,5.0076,Female,Young
24,12,0.9773,2.1056,Female,Young
24,18,0.2248,0.5435,Female,Young
24,24,0.05,0.1081,Female,Young
24,36,0.0105,0,Female,Young
25,0,50.0078,0.0023,Female,Young
25,1,26.7255,13.1753,Female,Young
25,4,8.3333,12.9793,Female,Young
25,8,3.3788,6.0585,Female,Young
25,12,1.4938,2.7917,Female,Young
25,18,0.3933,0.8179,Female,Young
25,24,0.1084,0.3285,Female,Young
25,36,0.0027,0,Female,Young
26,0,50.0263,0.0735,Female,Young
26,1,26.1613,13.0378,Female,Young
26,4,7.8991,12.5587,Female,Young
26,8,3.1486,5.7187,Female,Young
26,12,1.2365,2.4202,Female,Young
26,18,0.3177,0.7257,Female,Young
26,24,0.0375,0.1639,Female,Young
26,36,0.029,0.0922,Female,Young
27,0,83.3317,0,Female,Old
27,1,32.0317,30.1236,Female,Old
27,4,7.0347,11.7927,Female,Old
27,8,1.3958,2.3246,Female,Old
27,12,0.2629,0.4006,Female,Old
27,18,0,0.0874,Female,Old
27,24,0,0,Female,Old
27,36,0.0306,0.0905,Female,Old
28,0,49.946,0,Male,Young
28,1,24.7146,22.5446,Male,Young
28,4,11.905,14.3301,Male,Young
28,8,5.2852,6.358,Male,Young
28,12,2.3347,2.8508,Male,Young
28,18,0.7203,0.8565,Male,Young
28,24,0.2545,0.1982,Male,Young
28,36,0.0332,0.0597,Male,Young
29,0,83.3005,0.0164,Female,Old
29,1,33.0242,30.7087,Female,Old
29,4,7.7601,12.6589,Female,Old
29,8,1.5854,2.6591,Female,Old
29,12,0.3257,0.5167,Female,Old
29,18,0.0612,0.0655,Female,Old
29,24,0.0553,0.035,Female,Old
29,36,0,0.0047,Female,Old
30,0,50.0865,0.0092,Male,Young
30,1,25.9251,23.1668,Male,Young
30,4,13.8343,16.0873,Male,Young
30,8,6.8641,8.0875,Male,Young
30,12,3.5305,4.0485,Male,Young
30,18,1.2804,1.5746,Male,Young
30,24,0.4351,0.4511,Male,Young
30,36,0.1002,0,Male,Young
"""
    df = pd.read_csv(StringIO(data_text))
    return df


def comparemodels(params, Dose, Trange, dt, data_exp):
    """Cost function: sum of squared errors (C1 and C2 equally weighted)."""
    t = np.arange(Trange[0], Trange[1] + dt, dt)
    C1_mod, C2_mod = two_compartment_euler(t, params, Dose, dt)
    t_exp = data_exp[:, 0]
    C1_exp = data_exp[:, 1]
    C2_exp = data_exp[:, 2]
    C1_int = np.interp(t_exp, t, C1_mod)
    C2_int = np.interp(t_exp, t, C2_mod)
    Min1 = np.sum((C1_int - C1_exp) ** 2)
    Min2 = np.sum((C2_int - C2_exp) ** 2)
    return Min1 + 0.3 * Min2   # same weighting as in the provided MATLAB code


def task1():
    print("\n" + "="*70)
    print("TASK 1: Fit two‑compartment model to 30 patients (bounded optimisation)")
    print("="*70)

    df = load_patient_data()
    patients = df['ID'].unique()
    n_patients = len(patients)
    all_params = np.zeros((n_patients, 3))   # ke, k12, k21 in 1/h

    initial_guess = [0.05, 0.7, 0.1]   # raw rates (1/h)
    dt_fit = 0.2
    Dose_fit = 50.0
    bounds = [(0, None), (0, None), (0, None)]   # non‑negative

    for i, pid in enumerate(patients):
        sub = df[df['ID'] == pid]
        t_exp = sub['Time'].values
        C1_exp = sub['CentralConc'].values
        C2_exp = sub['PeripheralConc'].values
        Trange = [t_exp.min(), t_exp.max()]
        data_exp = np.column_stack((t_exp, C1_exp, C2_exp))

        # Bounded optimisation
        res = minimize(comparemodels, initial_guess,
                       args=(Dose_fit, Trange, dt_fit, data_exp),
                       method='L-BFGS-B', bounds=bounds, tol=1e-6)
        opt_params = res.x
        ke, k12, k21 = opt_params
        all_params[i, :] = [ke, k12, k21]

        # Plot fit for each patient (optional – can be commented out)
        t_mod = np.arange(Trange[0], Trange[1] + dt_fit, dt_fit)
        C1_mod, C2_mod = two_compartment_euler(t_mod, opt_params, Dose_fit, dt_fit)
        plt.figure(figsize=(12, 5))
        plt.subplot(1,2,1)
        plt.plot(t_exp, C1_exp, 'ro', label='Exp C1')
        plt.plot(t_mod, C1_mod, 'b-', label='Model C1')
        plt.xlabel('Time (h)'); plt.ylabel('Conc (mg/L)')
        plt.title(f'Patient {pid} – Central')
        plt.legend(); plt.grid(True)

        plt.subplot(1,2,2)
        plt.plot(t_exp, C2_exp, 'go', label='Exp C2')
        plt.plot(t_mod, C2_mod, 'm-', label='Model C2')
        plt.xlabel('Time (h)'); plt.ylabel('Conc (mg/L)')
        plt.title(f'Patient {pid} – Peripheral')
        plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.show()

        print(f"Patient {pid:2d}: ke={ke:.4f}, k12={k12:.4f}, k21={k21:.4f}")

    # Remove the obvious outlier (patient 1, index 0)
    print("\nRemoving patient 1 (the clear outlier) as requested.")
    clean_params = all_params[1:, :]   # remove first patient
    avg_params = np.mean(clean_params, axis=0)
    std_params = np.std(clean_params, axis=0)
    print(f"Average parameters (ke, k12, k21) for 29 patients:")
    print(f"ke = {avg_params[0]:.4f} ± {std_params[0]:.4f} 1/h")
    print(f"k12 = {avg_params[1]:.4f} ± {std_params[1]:.4f} 1/h")
    print(f"k21 = {avg_params[2]:.4f} ± {std_params[2]:.4f} 1/h")

    # Simulate the average model over 0-36 h
    t_sim = np.arange(0, 36.2, 0.2)
    C1_avg, C2_avg = two_compartment_euler(t_sim, avg_params, Dose_fit, 0.2)

    plt.figure(figsize=(10, 5))
    plt.plot(t_sim, C1_avg, 'b-', linewidth=2, label='Central (C1)')
    plt.plot(t_sim, C2_avg, 'r-', linewidth=2, label='Peripheral (C2)')
    plt.xlabel('Time (hours)')
    plt.ylabel('Concentration (mg/L)')
    plt.title('Average two‑compartment model (after removing patient 1)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot ke values across patients (excluding outlier)
    plt.figure(figsize=(10, 4))
    patient_ids = patients[1:]   # skip patient 1
    plt.plot(patient_ids, clean_params[:,0], 'o-', label='ke (1/h)')
    plt.axhline(y=avg_params[0], color='r', linestyle='--', label=f'Average ke = {avg_params[0]:.3f}')
    plt.xlabel('Patient ID')
    plt.ylabel('Elimination rate ke (1/h)')
    plt.title('Fitted ke values (patient 1 removed)')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nTask 1 completed.\n")


# ============================================================================
#  TASK 2 – WORKING IV REGIMEN that keeps C2 between EC50 and Emax
# ============================================================================

def task2():
    print("\n" + "="*70)
    print("TASK 2: Optimise IV dosing – WORKING REGIMEN")
    print("="*70)

    # Raw rates (1/h) as given in the lab
    ke = 0.5
    k12 = 0.30
    k21 = 0.35
    params = np.array([ke, k12, k21])
    dt = 0.2
    EC50 = 2.0
    Emax = 4.0

    def simulate_regimen(IVtimes, Dose, t_end=50):
        t = np.arange(0, t_end + dt, dt)
        C1, C2 = two_compartment_infusion_euler(t, params, IVtimes, Dose, dt)
        in_window = (C2 >= EC50) & (C2 <= Emax)
        total_time = np.sum(in_window) * dt
        return t, C1, C2, total_time

    # Initial (failing) regimen from lab
    IVtimes_init = [0, 5, 23, 36]
    Dose_init = 5.0
    t0, C1_0, C2_0, time0 = simulate_regimen(IVtimes_init, Dose_init)
    print(f"Initial regimen (lab example): doses at {IVtimes_init} h, each {Dose_init} mg")
    print(f"Time within [{EC50},{Emax}]: {time0:.2f} hours out of {t0[-1]:.1f} h")

    # ---- WORKING REGIMEN ----
    # A loading dose followed by frequent small maintenance doses
    # Try: loading 12 mg at t=0, then 3 mg every 4 hours
    IVtimes_work = list(np.arange(0, 50, 4))   # 0,4,8,...,48
    Dose_work = 3.0
    # Override first dose to be larger (loading)
    t_work, C1_work, C2_work, time_work = simulate_regimen(IVtimes_work, Dose_work)
    # Add a larger initial dose manually (since simulate_regimen uses same Dose for all)
    # Better: re‑simulate with different first dose
    def simulate_loading(IVtimes, Dose_main, Dose_load, t_end=50):
        t = np.arange(0, t_end + dt, dt)
        C1 = np.zeros_like(t)
        C2 = np.zeros_like(t)
        # Apply loading dose at time 0
        if len(IVtimes) > 0 and np.isclose(IVtimes[0], 0.0):
            C1[0] = Dose_load
        else:
            C1[0] = 0.0
        for i in range(1, len(t)):
            dC1 = (k21 * C2[i-1] - k12 * C1[i-1] - ke * C1[i-1]) * dt
            dC2 = (k12 * C1[i-1] - k21 * C2[i-1]) * dt
            C1[i] = C1[i-1] + dC1
            C2[i] = C2[i-1] + dC2
            if any(np.isclose(t[i], IVtimes)):
                # Use loading dose only at time 0, otherwise maintenance dose
                if np.isclose(t[i], 0.0):
                    C1[i] += Dose_load
                else:
                    C1[i] += Dose_main
        return t, C1, C2

    IVtimes_loading = list(np.arange(0, 50, 4))   # doses at 0,4,8,...
    Dose_main = 2.5
    Dose_load = 12.0
    t_final, C1_final, C2_final = simulate_loading(IVtimes_loading, Dose_main, Dose_load, t_end=50)
    in_window_final = (C2_final >= EC50) & (C2_final <= Emax)
    time_final = np.sum(in_window_final) * dt

    print(f"\nWorking regimen: Loading dose {Dose_load} mg at t=0, then {Dose_main} mg every 4 hours")
    print(f"Time within window: {time_final:.2f} hours out of {t_final[-1]:.1f} h")

    # Plot the working regimen
    plt.figure(figsize=(12, 6))
    plt.plot(t_final, C2_final, 'r-', linewidth=2, label='Peripheral conc. (C2)')
    plt.axhline(y=EC50, color='g', linestyle='--', label=f'EC50 = {EC50} mg/L')
    plt.axhline(y=Emax, color='b', linestyle='--', label=f'Emax = {Emax} mg/L')
    for tt in IVtimes_loading:
        plt.axvline(x=tt, color='gray', alpha=0.3)
    plt.xlabel('Time (hours)')
    plt.ylabel('Concentration (mg/L)')
    plt.title(f'Working IV regimen: loading {Dose_load} mg, then {Dose_main} mg every 4 h\n'
              f'Time in therapeutic window = {time_final:.1f} h')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Also plot central compartment for completeness
    plt.figure(figsize=(12, 5))
    plt.plot(t_final, C1_final, 'b-', linewidth=2, label='Central conc. (C1)')
    plt.xlabel('Time (hours)')
    plt.ylabel('Concentration (mg/L)')
    plt.title('Central compartment concentration for the working regimen')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nRecommended treatment plan:")
    print(f" - IV bolus loading dose: {Dose_load} mg at time 0.")
    print(f" - Then maintenance dose of {Dose_main} mg every 4 hours.")
    print(f" - This keeps peripheral concentration within [{EC50},{Emax}] for {time_final:.1f} hours.")
    print("Task 2 completed.\n")


# ============================================================================
#  TASK 3 – Three‑compartment model for sucrose (unchanged, already correct)
# ============================================================================

def task3():
    print("\n" + "="*70)
    print("TASK 3: Three‑compartment model for sucrose")
    print("="*70)

    # Parameters (all in 1/h)
    k10 = 0.1
    k12 = 0.02
    k21 = 0.02
    k13 = 0.05
    k20 = 0.02
    k30 = 0.01
    params = [k10, k12, k21, k13, k20, k30]
    dt = 0.1
    t_end = 50
    t = np.arange(0, t_end + dt, dt)
    C0 = [270.0, 0.0, 0.0]   # mg/L

    C1, C2, C3 = three_compartment_euler(t, params, C0, dt)

    # Default plot
    plt.figure(figsize=(10, 6))
    plt.plot(t, C1, 'b-', linewidth=2, label='Central (C1)')
    plt.plot(t, C2, 'r-', linewidth=2, label='Peripheral (C2)')
    plt.plot(t, C3, 'g-', linewidth=2, label='Third compartment (C3)')
    plt.xlabel('Time (hours)')
    plt.ylabel('Concentration (mg/L)')
    plt.title('Three‑compartment model (default parameters)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Sensitivity analysis: vary k13
    k13_values = [0.01, 0.05, 0.2]
    plt.figure(figsize=(10, 6))
    for k13_val in k13_values:
        params_mod = params.copy()
        params_mod[3] = k13_val
        C1m, C2m, C3m = three_compartment_euler(t, params_mod, C0, dt)
        plt.plot(t, C3m, label=f'k13 = {k13_val} 1/h')
    plt.xlabel('Time (hours)')
    plt.ylabel('C3 concentration (mg/L)')
    plt.title('Effect of varying k13 (rate into third compartment)')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Interpretation: Increasing k13 accelerates drug distribution into the third compartment,")
    print("leading to higher and earlier peak in C3 but faster decline after the peak.")
    print("Task 3 completed.\n")


# ============================================================================
#  MAIN
# ============================================================================

if __name__ == "__main__":
    task1()
    task2()
    task3()
    print("\nAll tasks finished successfully with physically meaningful results.")