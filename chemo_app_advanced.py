import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def chemo_equilibrium(r, K, alpha, beta, gamma, n0, D0, cycles=30):
    n_values = [n0]
    D_values = [D0]
    for _ in range(cycles):
        n_new = n_values[-1] * np.exp(r * (1 - n_values[-1] / K) - alpha * D_values[-1])
        D_new = beta * D_values[-1] + gamma * n_values[-1]
        n_values.append(n_new)
        D_values.append(D_new)
    df = pd.DataFrame({
        'Cycle': range(cycles+1), 
        'TumorDensity_n': n_values, 
        'DrugConcentration_D': D_values
    })
    return df

st.title("Chemotherapy Equilibrium Model – Advanced Analysis (with Dynamic Overlay)")

# Main parameters
r = st.number_input("Tumor growth rate (r)", value=0.2)
K = st.number_input("Carrying capacity (K)", value=1.0)
alpha = st.number_input("Drug cytotoxicity (alpha)", value=0.7)
beta = st.number_input("Drug retention (beta)", value=0.5)
gamma = st.number_input("Feedback parameter (gamma)", value=0.1)
n0 = st.number_input("Initial tumor cell density (n0)", value=0.5)
D0 = st.number_input("Initial drug level (D0)", value=1.0)
cycles = st.slider("Number of chemotherapy cycles", 5, 60, 30)

if st.button("Run Simulation"):
    df_equilibrium = chemo_equilibrium(r, K, alpha, beta, gamma, n0, D0, cycles)
    
    st.subheader("Last 5 Simulation Results")
    st.dataframe(df_equilibrium.tail())
    
    # CSV export feature
    csv = df_equilibrium.to_csv(index=False)
    st.download_button("Download results as CSV", csv, "chemo_simulation.csv")
    
    # 1. Time-series plot
    st.subheader("Time-series of Tumor Density and Drug Concentration")
    fig1, ax1 = plt.subplots()
    ax1.plot(df_equilibrium['Cycle'], df_equilibrium['TumorDensity_n'], label='Tumor Density (n)')
    ax1.plot(df_equilibrium['Cycle'], df_equilibrium['DrugConcentration_D'], label='Drug Concentration (D)')
    ax1.set_xlabel('Chemotherapy Cycle')
    ax1.set_ylabel('Normalized value')
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)
    
    # 2. Dynamic overlay plot: tumor density for a range of alphas
    st.subheader("Overlay: Tumor Density for selected alpha values (dynamic, set by sliders)")
    a_min, a_max = st.slider("Alpha overlay range", 0.1, 1.2, (0.2, 1.0))
    a_steps = st.slider("Number of overlay curves", 2, 10, 4)
    overlay_alphas = np.linspace(a_min, a_max, a_steps)
    fig2, ax2 = plt.subplots()
    for a in overlay_alphas:
        df = chemo_equilibrium(r, K, a, beta, gamma, n0, D0, cycles)
        label = f'alpha={a:.2f}'
        ax2.plot(df['Cycle'], df['TumorDensity_n'], label=label)
    ax2.set_xlabel('Cycle')
    ax2.set_ylabel('Tumor Density')
    ax2.legend()
    ax2.set_title('Overlay: Tumor Density vs Cycles (dynamic alphas)')
    ax2.grid(True)
    st.pyplot(fig2)
    
    # 3. Two-parameter sweep heatmap (alpha, beta)
    st.subheader("Heatmap: Final Tumor Density (alpha vs beta)")
    a_min_hm, a_max_hm = st.slider("Alpha range for heatmap", 0.1, 1.2, (0.2, 1.0))
    b_min_hm, b_max_hm = st.slider("Beta range for heatmap", 0.1, 1.0, (0.2, 0.8))
    nsteps = st.slider("Heatmap grid resolution", 5, 20, 10)
    alphas = np.linspace(a_min_hm, a_max_hm, nsteps)
    betas = np.linspace(b_min_hm, b_max_hm, nsteps)
    final_n_matrix = np.zeros((len(betas), len(alphas)))
    for i, b in enumerate(betas):
        for j, a in enumerate(alphas):
            df = chemo_equilibrium(r, K, a, b, gamma, n0, D0, cycles)
            final_n_matrix[i, j] = df['TumorDensity_n'].iloc[-1]
    fig3, ax3 = plt.subplots()
    im = ax3.imshow(final_n_matrix, origin='lower', 
                     extent=[alphas[0], alphas[-1], betas[0], betas[-1]],
                     aspect='auto', cmap='viridis')
    fig3.colorbar(im, ax=ax3, label="Final Tumor Density")
    ax3.set_xlabel("Drug cytotoxicity (alpha)")
    ax3.set_ylabel("Drug retention (beta)")
    ax3.set_title(f"Heatmap: Final Tumor Density after {cycles} cycles")
    st.pyplot(fig3)
