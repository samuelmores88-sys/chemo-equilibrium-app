import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def chemoequilibrium(r, K, alpha, beta, gamma, n0, D0, cycles=30):
    nvalues = [n0]
    Dvalues = [D0]
    for _ in range(cycles):
        nnew = nvalues[-1] * np.exp(r * (1 - nvalues[-1] / K) - alpha * Dvalues[-1])
        Dnew = beta * Dvalues[-1] + gamma * nvalues[-1]
        nvalues.append(nnew)
        Dvalues.append(Dnew)
    df = pd.DataFrame({
        'Cycle': range(cycles + 1),
        'Tumor Density (n)': nvalues,
        'Drug Concentration (D)': Dvalues
    })
    return df

st.title("Chemotherapy Equilibrium Model – Advanced Analysis (with Dynamic Overlay)")

r = st.number_input("Tumor growth rate (r)", value=0.2)
K = st.number_input("Carrying capacity (K)", value=1.0)
alpha = st.number_input("Drug cytotoxicity (alpha)", value=0.7)
beta = st.number_input("Drug retention (beta)", value=0.5)
gamma = st.number_input("Feedback parameter (gamma)", value=0.1)
n0 = st.number_input("Initial tumor cell density (n0)", value=0.5)
D0 = st.number_input("Initial drug level (D0)", value=1.0)
cycles = st.slider("Number of chemotherapy cycles", 5, 60, 30)

if st.button("Run Simulation"):
    df_equilibrium = chemoequilibrium(r, K, alpha, beta, gamma, n0, D0, cycles)
    
    st.subheader("Last 5 Simulation Results")
    st.dataframe(df_equilibrium.tail())
    
    csv = df_equilibrium.to_csv(index=False)
    st.download_button("Download results as CSV", csv, "chemosimulation.csv")
    
    st.subheader("Time-series of Tumor Density and Drug Concentration")
    fig1, ax1 = plt.subplots()
    ax1.plot(df_equilibrium['Cycle'], df_equilibrium['Tumor Density (n)'], label="Tumor Density (n)")
    ax1.plot(df_equilibrium['Cycle'], df_equilibrium['Drug Concentration (D)'], label="Drug Concentration (D)")
    ax1.set_xlabel("Chemotherapy Cycle")
    ax1.set_ylabel("Normalized value")
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)
    
    st.subheader("Overlay Tumor Density for selected alpha values")
    amin, amax = st.slider("Alpha overlay range", 0.1, 1.2, (0.2, 1.0))
    asteps = st.slider("Number of overlay curves", 2, 10, 4)
    overlay_alphas = np.linspace(amin, amax, asteps)
    
    fig2, ax2 = plt.subplots()
    for a in overlay_alphas:
        df = chemoequilibrium(r, K, a, beta, gamma, n0, D0, cycles)
        label = f"α={a:.2f}"
        ax2.plot(df['Cycle'], df['Tumor Density (n)'], label=label)
    ax2.set_xlabel("Cycle")
    ax2.set_ylabel("Tumor Density")
    ax2.legend()
    ax2.set_title("Overlay Tumor Density vs Cycles (dynamic alphas)")
    ax2.grid(True)
    st.pyplot(fig2)
    
    st.subheader("Heatmap: Final Tumor Density (alpha vs beta)")
    aminhm, amaxhm = st.slider("Alpha range for heatmap", 0.1, 1.2, (0.2, 1.0))
    bminhm, bmaxhm = st.slider("Beta range for heatmap", 0.1, 1.0, (0.2, 0.8))
    nsteps = st.slider("Heatmap grid resolution", 5, 20, 10)
    
    alphas = np.linspace(aminhm, amaxhm, nsteps)
    betas = np.linspace(bminhm, bmaxhm, nsteps)
    final_n_matrix = np.zeros((len(betas), len(alphas)))
    
    for i, b in enumerate(betas):
        for j, a in enumerate(alphas):
            df = chemoequilibrium(r, K, a, b, gamma, n0, D0, cycles)
            final_n_matrix[i, j] = df['Tumor Density (n)'].iloc[-1]
    
    fig3, ax3 = plt.subplots()
    im = ax3.imshow(final_n_matrix, origin='lower', extent=[alphas[0], alphas[-1], betas[0], betas[-1]], aspect='auto', cmap='viridis')
    fig3.colorbar(im, ax=ax3, label='Final Tumor Density')
    ax3.set_xlabel("Drug cytotoxicity (alpha)")
    ax3.set_ylabel("Drug retention (beta)")
    ax3.set_title(f"Heatmap: Final Tumor Density after {cycles} cycles")
    st.pyplot(fig3)

# Add footer with author name
st.markdown("---")
st.markdown(
    "<div style='position: fixed; bottom: 10px; right: 10px; font-size: 12px; color: gray;'>"
    "Durga Rao Pathuri"
    "</div>",
    unsafe_allow_html=True
)
