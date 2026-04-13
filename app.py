"""
app.py  -  P452 Quantum Simulator  |  Streamlit frontend
Run with:  streamlit run app.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from backend import (
    run_circuit,
    get_statevector,
    build_ry_cnot_circuit,
    build_ghz_circuit,
    build_superposition_201_425,
    build_cnot_chain_circuit,
    build_reverse_cnot_chain_circuit,
    build_teleportation_circuit,
    build_long_distance_cnot,
    TOTAL_QUBITS,
)
from hubbard import build_trotter_step, evolve, rabi_transfer

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="P452 Quantum Simulator",
    page_icon="⚛️",
    layout="wide",
)

st.title("⚛️  P452 – Universal Quantum Computer Simulator")
st.caption("10-qubit backend powered by Qiskit Aer · Streamlit UI")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def draw_circuit(qc, fold=30):
    fig = qc.draw(output="mpl", fold=fold, style={"backgroundcolor": "#FFFFFF"})
    return fig

def plot_histogram(counts, title=""):
    total = sum(counts.values())
    labels = sorted(counts.keys())
    probs  = [counts[k] / total for k in labels]
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 0.7), 4))
    bars = ax.bar(labels, probs, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Measurement outcome", fontsize=11)
    ax.set_ylabel("Probability", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.15)
    for bar, p in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width()/2, p + 0.02,
                f"{p:.2f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR  –  main controls (matches the project requirement exactly)
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Simulator Controls")

# 1. Preset selector
preset = st.sidebar.selectbox(
    "Select Preset Circuit",
    [
        "Teleportation",
        "Hubbard Model",
        "── Checkpoints ──",
        "Q1.2 – Ry(θ)–CNOT",
        "Q1.3 – 10-Qubit GHZ",
        "Q1.4 – Unitarity & State Recovery",
        "Q2.2 – Long-Distance CNOT",
        "Q2.3 – Teleportation Statistics",
        "Q3.1 – Trotter Circuit",
        "Q3.2 – Non-Interacting Dynamics",
        "Q3.3 – Mott Insulator Physics",
    ],
)

# 2. Parameter slider (changes meaning based on preset)
st.sidebar.markdown("---")
if preset == "Teleportation" or preset == "Q1.2 – Ry(θ)–CNOT" or preset == "Q2.3 – Teleportation Statistics":
    theta = st.sidebar.slider("Rotation angle θ (rad)", 0.0, float(2*np.pi), float(np.pi), 0.01)
    st.sidebar.latex(r"\theta = " + f"{theta:.3f}" + r"\ \text{rad} \approx " + f"{theta/np.pi:.3f}" + r"\pi")
elif preset in ("Hubbard Model", "Q3.1 – Trotter Circuit", "Q3.2 – Non-Interacting Dynamics", "Q3.3 – Mott Insulator Physics"):
    J_val = st.sidebar.slider("Hopping amplitude J", 0.1, 3.0, 1.0, 0.1)
    U_val = st.sidebar.slider("On-site interaction U", 0.0, 20.0, 0.0 if preset == "Q3.2 – Non-Interacting Dynamics" else 5.0, 0.5)
    st.sidebar.latex(r"U/J = " + f"{U_val/J_val:.2f}")
elif preset == "Q2.2 – Long-Distance CNOT":
    target_q = st.sidebar.slider("Target qubit (control = q0)", 2, 4, 4, 1)

shots = st.sidebar.slider("Shots", 256, 4096, 1024, 256)
st.sidebar.markdown("---")
st.sidebar.info("Use the **Preset** dropdown to switch circuits. Sliders update parameters in real time.")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN PANEL
# ─────────────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════
#  PRESET: TELEPORTATION  (Q2.1)
# ══════════════════════════════════════════════════════════════
if preset == "Teleportation":
    st.header("🔀 Quantum Teleportation")
    st.markdown(r"""
Alice teleports $|q_0\rangle = \frac{1}{\sqrt{5}}(2|0\rangle + |1\rangle)$ to Bob
using a shared Bell pair. Adjust **θ** in the sidebar to change Alice's input state.
    """)

    alpha = np.cos(theta / 2)
    beta  = np.sin(theta / 2)
    st.markdown(f"**Current state:** α = `{alpha:.3f}`, β = `{beta:.3f}`")

    qc_tp = build_teleportation_circuit(alpha=alpha, beta=beta)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Circuit Diagram")
        fig_c = draw_circuit(qc_tp, fold=30)
        st.pyplot(fig_c); plt.close(fig_c)

    with col2:
        st.subheader("Stage Labels")
        st.markdown("""
| Stage | Gates |
|-------|-------|
| Alice state ready | Ry(θ) on q0 |
| Bell State Preparation | H + CNOT on q1,q2 |
| Bell Measurement | CNOT + H on q0,q1 |
| Classical corrections | X/Z on q2 conditioned on c0,c1 |
        """)
        st.subheader("Run Simulation")
        if st.button("▶  Run"):
            counts = run_circuit(qc_tp, shots=shots)
            fig_h = plot_histogram(counts, title=f"Teleportation  |  {shots} shots")
            st.pyplot(fig_h); plt.close(fig_h)

# ══════════════════════════════════════════════════════════════
#  PRESET: HUBBARD MODEL  (Q3)
# ══════════════════════════════════════════════════════════════
elif preset == "Hubbard Model":
    st.header("⚛️ Fermi-Hubbard Model")
    st.markdown(rf"""
2-site dimer with $J = {J_val}$, $U = {U_val}$.  
Adjust **J** and **U** in the sidebar. Select an initial state below and run the time evolution.
    """)

    init_state = st.selectbox(
        "Initial state",
        ["1000 — one ↑ electron at Site 1",
         "1100 — both electrons at Site 1",
         "1001 — ↑ at Site 1, ↓ at Site 2"],
    )
    init_bits = init_state.split(" ")[0]

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("One Trotter Step Circuit")
        dt_show = st.slider("δt for diagram", 0.05, 0.5, 0.1, 0.05)
        qc_step = build_trotter_step(J=J_val, U=U_val, dt=dt_show)
        fig_tr = draw_circuit(qc_step, fold=40)
        st.pyplot(fig_tr); plt.close(fig_tr)

    with col2:
        st.subheader("Gate Legend")
        st.markdown(r"""
| Gate | Term | Physics |
|------|------|---------|
| RXX + RYY | $\frac{J}{2}(XX+YY)$ | Electron hopping |
| CZ (bracket) | Z-string | Fermionic anti-commutation |
| RZ | $-\frac{U}{4}Z$ | On-site energy shift |
| RZZ | $\frac{U}{4}ZZ$ | Coulomb repulsion |
        """)

    st.subheader("Time Evolution")
    n_pts   = st.slider("Time points", 20, 80, 40, 10)
    n_steps = st.slider("Trotter steps per point", 5, 30, 15, 5)

    if st.button("▶  Run Time Evolution"):
        with st.spinner("Simulating..."):
            times, probs = evolve(
                initial_state=init_bits,
                J=J_val, U=U_val,
                tau_max=np.pi,
                n_time_points=n_pts,
                n_trotter_steps=n_steps,
            )

        # Show top 4 most populated states
        avg_pop = {k: np.mean(v) for k, v in probs.items()}
        top4 = sorted(avg_pop, key=avg_pop.get, reverse=True)[:4]

        colors = ["#4C72B0", "#DD8452", "#2ca02c", "#9467bd"]
        fig_ev, ax = plt.subplots(figsize=(8, 4))
        for state, color in zip(top4, colors):
            ax.plot(times, probs[state], "o-", color=color,
                    markersize=3, label=f"|{state}⟩")
        ax.set_xlabel(r"Time $\tau$", fontsize=12)
        ax.set_ylabel("Probability", fontsize=12)
        ax.set_title(rf"Hubbard Dynamics: $J={J_val}$, $U={U_val}$, init=$|{init_bits}\rangle$", fontsize=13)
        ax.legend(); ax.set_ylim(-0.05, 1.15)
        ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        ax.set_xticklabels(["0","π/4","π/2","3π/4","π"])
        plt.tight_layout()
        st.pyplot(fig_ev); plt.close(fig_ev)

# ══════════════════════════════════════════════════════════════
#  CHECKPOINTS
# ══════════════════════════════════════════════════════════════
elif preset == "── Checkpoints ──":
    st.info("👈 Select a checkpoint from the sidebar dropdown.")

elif preset == "Q1.2 – Ry(θ)–CNOT":
    st.header("Q1.2 – Parameter Control Loop")
    st.markdown(r"Circuit: $R_y(\theta)$ on $q_0$, then CNOT $q_0 \to q_1$.")
    qc = build_ry_cnot_circuit(theta)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Circuit")
        st.pyplot(draw_circuit(qc, fold=-1)); plt.close()
    with col2:
        st.subheader("Histogram")
        counts = run_circuit(qc, shots=shots)
        st.pyplot(plot_histogram(counts, f"Ry(θ={theta:.2f})–CNOT")); plt.close()
    st.markdown(r"""
**Logic Check:** $R_y(\pi)|0\rangle = |1\rangle$, so CNOT flips $q_1$ → outcome $|11\rangle$ with ~100%.
This proves the backend received the exact slider value.
    """)

elif preset == "Q1.3 – 10-Qubit GHZ":
    st.header("Q1.3 – 10-Qubit GHZ State")
    st.latex(r"|GHZ\rangle = \frac{|0\rangle + |1023\rangle}{\sqrt{2}}")
    qc_ghz = build_ghz_circuit()
    st.subheader("Circuit Diagram")
    st.pyplot(draw_circuit(qc_ghz, fold=40)); plt.close()
    if st.button("▶  Run"):
        counts = run_circuit(qc_ghz, shots=shots)
        st.pyplot(plot_histogram(counts, f"10-Qubit GHZ | {shots} shots")); plt.close()
        st.info("Only |0000000000⟩ and |1111111111⟩ appear — confirming 10-qubit entanglement.")

elif preset == "Q1.4 – Unitarity & State Recovery":
    st.header("Q1.4 – Unitarity & State Recovery")
    tab1, tab2, tab3 = st.tabs(["Step 1 – Prepare", "Step 2 – CNOT Chain", "Step 3 – Recover"])

    def show_sv(qc):
        sv = get_statevector(qc)
        rows = [(f"|{format(i,'010b')}⟩ (={i})", amp, abs(amp)**2)
                for i, amp in enumerate(sv) if abs(amp) > 1e-6]
        for label, amp, prob in rows:
            st.write(f"**{label}** → amplitude `{amp:.4f}`, prob `{prob:.4f}`")

    with tab1:
        qc1 = build_superposition_201_425()
        st.pyplot(draw_circuit(qc1, fold=30)); plt.close()
        if st.button("Compute statevector – Step 1"): show_sv(qc1)
    with tab2:
        qc2 = build_cnot_chain_circuit()
        st.pyplot(draw_circuit(qc2, fold=30)); plt.close()
        if st.button("Compute statevector – Step 2"): show_sv(qc2)
    with tab3:
        qc3 = build_reverse_cnot_chain_circuit()
        st.pyplot(draw_circuit(qc3, fold=30)); plt.close()
        if st.button("Compute statevector – Step 3"): show_sv(qc3)
    st.markdown(r"""
**Unitarity:** Each CNOT is its own inverse. Reversing the chain recovers the original state
$\frac{1}{\sqrt{2}}(|201\rangle+|425\rangle)$, confirming $U^\dagger U = I$.
    """)

elif preset == "Q2.2 – Long-Distance CNOT":
    st.header("Q2.2 – Long-Distance CNOT via SWAP Chain")
    qc_ld = build_long_distance_cnot(control=0, target=target_q)
    st.pyplot(draw_circuit(qc_ld, fold=-1)); plt.close()
    n_swaps = (target_q - 1) * 2
    total   = n_swaps * 3 + 1
    st.markdown(f"""
| Metric | Value |
|--------|-------|
| Distance | {target_q} qubits apart |
| SWAP gates | {n_swaps} |
| CNOTs per SWAP | 3 |
| Core CNOT | 1 |
| **Total CNOTs** | **{total}** |
    """)

elif preset == "Q2.3 – Teleportation Statistics":
    st.header("Q2.3 – Teleportation Statistics")
    st.markdown("Alice teleports $|0\\rangle$ to Bob. Running 1024 shots.")
    if st.button("▶  Run"):
        from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit as QC
        qr = QuantumRegister(3, "q")
        ca = ClassicalRegister(2, "ca")
        cb = ClassicalRegister(1, "cb")
        qc_s = QC(qr, ca, cb)
        qc_s.h(qr[1]); qc_s.cx(qr[1], qr[2])
        qc_s.cx(qr[0], qr[1]); qc_s.h(qr[0])
        qc_s.measure(qr[0], ca[0]); qc_s.measure(qr[1], ca[1])
        qc_s.x(qr[2]).c_if(ca[1], 1)
        qc_s.z(qr[2]).c_if(ca[0], 1)
        qc_s.measure(qr[2], cb[0])
        from backend import simulator
        counts = simulator.run(qc_s, shots=shots).result().get_counts()
        b0 = sum(v for k, v in counts.items() if k.split(" ")[0] == "0")
        b1 = sum(v for k, v in counts.items() if k.split(" ")[0] == "1")
        total = b0 + b1
        col1, col2 = st.columns(2)
        col1.metric("P(Bob |0⟩)", f"{b0/total*100:.1f}%")
        col2.metric("P(Bob |1⟩)", f"{b1/total*100:.1f}%")
        fig, ax = plt.subplots(figsize=(4,3))
        ax.bar(["Bob |0⟩","Bob |1⟩"],[b0/total,b1/total],color=["#4C72B0","#DD8452"])
        ax.set_ylim(0,1.1); ax.set_ylabel("Probability")
        ax.set_title(f"Bob's outcome | {shots} shots")
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown(r"**Expected:** ~100% in $|0\rangle$. Any deviation is shot noise $\sim 1/\sqrt{N}$.")

elif preset == "Q3.1 – Trotter Circuit":
    st.header("Q3.1 – Trotter Step Circuit")
    qc_tr = build_trotter_step(J=J_val, U=U_val, dt=0.1)
    st.pyplot(draw_circuit(qc_tr, fold=40)); plt.close()
    st.markdown(r"""
| Block | Gate | Jordan-Wigner origin |
|-------|------|---------------------|
| Hopping ↑ (q0↔q2) | CZ·RXX·RYY·CZ | $\frac{J}{2}(XX+YY)$ + Z-string |
| Hopping ↓ (q1↔q3) | CZ·RXX·RYY·CZ | $\frac{J}{2}(XX+YY)$ + Z-string |
| Interaction site 1 | RZ + RZZ on (q0,q1) | $\frac{U}{4}(I-Z_\uparrow-Z_\downarrow+Z_\uparrow Z_\downarrow)$ |
| Interaction site 2 | RZ + RZZ on (q2,q3) | same |
    """)

elif preset == "Q3.2 – Non-Interacting Dynamics":
    st.header("Q3.2 – Non-Interacting Dynamics (U = 0)")
    st.markdown(r"$U=0$, $J=1$. Initial state $|1000\rangle$. Plot $P(|0010\rangle)$ vs $\tau$.")
    n_pts = st.slider("Time points", 20, 80, 40, 10)
    n_stp = st.slider("Trotter steps", 5, 40, 20, 5)
    if st.button("▶  Run"):
        with st.spinner("Simulating..."):
            times, probs = evolve("1000", J=1.0, U=0.0,
                                  tau_max=np.pi, n_time_points=n_pts, n_trotter_steps=n_stp)
        analytical = rabi_transfer(1.0, times)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(times, probs["0010"], "o-", color="#4C72B0", markersize=4, label=r"Sim: $P(|0010\rangle)$")
        ax.plot(times, probs["1000"], "s--", color="#DD8452", markersize=4, label=r"Sim: $P(|1000\rangle)$")
        ax.plot(times, analytical, "-", color="red", lw=2, alpha=0.7, label=r"Analytical: $\sin^2(J\tau)$")
        ax.axvline(np.pi/2, color="gray", ls=":", label=r"$\tau=\pi/2$")
        ax.set_xlabel(r"Time $\tau$"); ax.set_ylabel("Probability")
        ax.set_title(r"Non-Interacting Dynamics: $U=0$, $J=1$")
        ax.legend(); ax.set_xlim(0, np.pi)
        ax.set_xticks([0,np.pi/4,np.pi/2,3*np.pi/4,np.pi])
        ax.set_xticklabels(["0","π/4","π/2","3π/4","π"])
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown(r"""
**Transfer complete at** $\tau^* = \pi/(2J) = \pi/2$.  
Matches Rabi oscillation $P(\tau)=\sin^2(J\tau)$ of a two-level system with coupling $J$.
        """)

elif preset == "Q3.3 – Mott Insulator Physics":
    st.header("Q3.3 – Strong Interactions & Mott Physics")
    st.markdown(r"$U=10$, $J=1$. Initial state $|1100\rangle$ (both electrons at Site 1).")
    n_pts = st.slider("Time points", 20, 80, 40, 10)
    n_stp = st.slider("Trotter steps", 5, 40, 20, 5)
    if st.button("▶  Run"):
        with st.spinner("Simulating interacting..."):
            times, probs = evolve("1100", J=1.0, U=U_val,
                                  tau_max=np.pi, n_time_points=n_pts, n_trotter_steps=n_stp)
        with st.spinner("Simulating free reference..."):
            _, probs0 = evolve("1100", J=1.0, U=0.0,
                               tau_max=np.pi, n_time_points=n_pts, n_trotter_steps=n_stp)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, p, title in zip(axes, [probs, probs0],
                                [rf"Interacting $U={U_val}$, $J=1$",
                                 r"Free $U=0$, $J=1$ (reference)"]):
            ax.plot(times, p["1100"], "o-", color="#4C72B0", markersize=3, label=r"$P(|1100\rangle)$")
            ax.plot(times, p["0011"], "s-", color="#DD8452", markersize=3, label=r"$P(|0011\rangle)$ doublon")
            ax.set_xlabel(r"Time $\tau$"); ax.set_ylabel("Probability")
            ax.set_title(title); ax.legend()
            ax.set_xlim(0, np.pi); ax.set_ylim(-0.05, 1.15)
            ax.set_xticks([0,np.pi/2,np.pi]); ax.set_xticklabels(["0","π/2","π"])
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown(rf"""
**Mott Insulator:** At $U/J={U_val:.0f}$, tunnelling is suppressed — $P(|1100\rangle) \approx 1$.  
Residual oscillation amplitude $\sim (J/U)^2 = {(1/U_val)**2:.4f}$.  
This is the quantum signature of a Mott insulating state: Coulomb repulsion localises electrons.
        """)
