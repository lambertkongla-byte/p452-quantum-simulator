# P452 – Quantum Computer Simulator

A full-stack 10-qubit quantum simulator built with **Qiskit Aer** (backend) and **Streamlit** (frontend).

---

## Quick Start

### 1. Clone / download the project
```
your_folder/
├── app.py
├── backend.py
└── requirements.txt
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```
The app opens automatically at **http://localhost:8501**

---

## Project Structure

| File | Purpose |
|------|---------|
| `backend.py` | All quantum circuit builders + AerSimulator runner |
| `app.py` | Streamlit UI – links sliders/buttons to the backend |
| `requirements.txt` | Pinned Python dependencies |

---

## Checkpoint Coverage

| Question | Section in app |
|----------|---------------|
| Q1.2 | Parameter Control – Ry(θ)–CNOT demo |
| Q1.3 | 10-Qubit GHZ state circuit & histogram |
| Q1.4 | Unitarity: prepare → transform → recover |
| Q2.1 | 3-qubit teleportation circuit diagram |
| Q2.2 | Long-distance CNOT via SWAP chain |
| Q2.3 | 1024-shot teleportation statistics |

Phase 3 (Fermi-Hubbard) is in a separate file `hubbard.py` (coming next).

---

## Deploying to Streamlit Cloud

1. Push your folder to a **public GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. Set **Main file path** to `app.py`.
4. Click **Deploy** — the live URL will be shared automatically.
