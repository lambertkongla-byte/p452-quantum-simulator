# ⚛️ P452 – Universal Quantum Computer Simulator

A full-stack 10-qubit quantum simulator built for PHYS 452 at the University of Chicago.

**Live App:** [Click to open](https://lambertkongla-byte-p452-quantum-simulator-app.streamlit.app](https://p452-quantum-simulator-nu6fukmu2jbfn53a5bgmbe.streamlit.app/)  
**GitHub:** [lambertkongla-byte/p452-quantum-simulator](https://github.com/lambertkongla-byte/p452-quantum-simulator)

---

## Overview

This project implements a universal quantum computer simulator using:
- **Backend:** Qiskit Aer (`AerSimulator`) — 10-qubit statevector + shot-based simulation
- **Frontend:** Streamlit — interactive web UI with preset circuits, parameter sliders, circuit diagrams, and measurement histograms

---

## Features

### Preset Circuits (Sidebar)
| Preset | Description |
|--------|-------------|
| **Teleportation** | 3-qubit quantum teleportation with adjustable input state θ |
| **Hubbard Model** | 2-site Fermi-Hubbard time evolution with adjustable J and U |

### Checkpoint Questions Covered
| Question | Topic |
|----------|-------|
| Q1.2 | Parameter control: Ry(θ)–CNOT circuit with live histogram |
| Q1.3 | 10-qubit GHZ state circuit and measurement |
| Q1.4 | Unitarity: prepare → CNOT chain → reverse recovery |
| Q2.1 | Quantum teleportation circuit with Bell state labels |
| Q2.2 | Long-distance CNOT via SWAP chain with gate count |
| Q2.3 | 1024-shot teleportation statistics |
| Q3.1 | Trotter step circuit with Jordan-Wigner gate labels |
| Q3.2 | Non-interacting dynamics: P(\|0010⟩) vs τ with Rabi comparison |
| Q3.3 | Mott insulator physics: suppressed tunnelling at large U/J |

---

## Project Structure

```
p452-quantum-simulator/
├── app.py            # Streamlit frontend
├── backend.py        # 10-qubit AerSimulator + circuit builders (Phase 1 & 2)
├── hubbard.py        # Fermi-Hubbard Trotterization + time evolution (Phase 3)
├── requirements.txt  # Python dependencies
└── README.md
```

---

## Physics Summary

### Phase 1 & 2 — Quantum Circuits
- 10-qubit GHZ state: $(|0\rangle + |1023\rangle)/\sqrt{2}$
- Quantum teleportation using Bell pairs and classical feed-forward
- Unitarity demonstrated via CNOT chain + inverse recovery

### Phase 3 — Fermi-Hubbard Model
The 2-site Hubbard Hamiltonian is:

$$H = -J\sum_\sigma(c^\dagger_{1\sigma}c_{2\sigma} + \text{h.c.}) + U\sum_i n_{i\uparrow}n_{i\downarrow}$$

Mapped to qubits via **Jordan-Wigner transformation**:
- Hopping term → $\frac{J}{2}(XX + YY)$ → RXX + RYY gates
- Interaction term → $\frac{U}{4}(I - Z_\uparrow - Z_\downarrow + Z_\uparrow Z_\downarrow)$ → RZ + RZZ gates

Time evolution $e^{-iH\tau}$ is implemented via **first-order Trotterization**.

---

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/lambertkongla-byte/p452-quantum-simulator.git
cd p452-quantum-simulator

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

Opens at **http://localhost:8501**

---

## Deployment

Deployed via [Streamlit Cloud](https://share.streamlit.io).  
Any push to `main` automatically redeploys the live app.

---

*PHYS 452 — Project 1 | University of Chicago | Spring 2026*
