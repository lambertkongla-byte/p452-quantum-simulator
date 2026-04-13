"""
backend.py  -  10-qubit Qiskit-Aer simulator backend (Qiskit 2.x compatible)
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

simulator = AerSimulator()
TOTAL_QUBITS = 10


def run_circuit(qc: QuantumCircuit, shots: int = 1024) -> dict:
    if qc.num_clbits == 0:
        qc = qc.copy()
        qc.measure_all()
    job = simulator.run(qc, shots=shots)
    return job.result().get_counts()


def get_statevector(qc: QuantumCircuit) -> np.ndarray:
    qc2 = qc.copy()
    qc2.save_statevector()
    job = simulator.run(qc2)
    return np.array(job.result().get_statevector())


# Q1.2 - Ry(theta) + CNOT
def build_ry_cnot_circuit(theta: float) -> QuantumCircuit:
    qc = QuantumCircuit(2, 2)
    qc.ry(theta, 0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])
    return qc


# Q1.3 - 10-qubit GHZ
def build_ghz_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(TOTAL_QUBITS, TOTAL_QUBITS)
    qc.h(0)
    for i in range(TOTAL_QUBITS - 1):
        qc.cx(i, i + 1)
    qc.measure(range(TOTAL_QUBITS), range(TOTAL_QUBITS))
    return qc


# Q1.4 - Superposition of |201> and |425>
def build_superposition_201_425() -> QuantumCircuit:
    n = TOTAL_QUBITS
    val_a, val_b = 201, 425
    bits_a = format(val_a, f"0{n}b")
    bits_b = format(val_b, f"0{n}b")

    def qubit_idx(pos):
        return n - 1 - pos

    common_ones, diff_a_only, diff_b_only = [], [], []
    for pos in range(n):
        q = qubit_idx(pos)
        ba, bb = bits_a[pos], bits_b[pos]
        if ba == "1" and bb == "1":
            common_ones.append(q)
        elif ba == "1" and bb == "0":
            diff_a_only.append(q)
        elif ba == "0" and bb == "1":
            diff_b_only.append(q)

    qc = QuantumCircuit(n)
    for pos in range(n):
        if bits_a[pos] == "1":
            qc.x(qubit_idx(pos))
    qc.barrier()

    ctrl_q = diff_b_only[0] if diff_b_only else diff_a_only[0]
    if ctrl_q in diff_a_only:
        qc.x(ctrl_q)
    qc.h(ctrl_q)
    qc.barrier()

    for q in diff_a_only:
        if q != ctrl_q:
            qc.cx(ctrl_q, q)
    for q in diff_b_only:
        if q != ctrl_q:
            qc.cx(ctrl_q, q)
    qc.barrier()
    return qc


def build_cnot_chain_circuit() -> QuantumCircuit:
    qc = build_superposition_201_425()
    qc.barrier()
    for i in range(TOTAL_QUBITS - 1):
        qc.cx(i, i + 1)
    return qc


def build_reverse_cnot_chain_circuit() -> QuantumCircuit:
    qc = build_cnot_chain_circuit()
    qc.barrier()
    for i in range(TOTAL_QUBITS - 2, -1, -1):
        qc.cx(i, i + 1)
    return qc


# Q2.1 - Teleportation (c_if compatible with Qiskit 2.x)
def build_teleportation_circuit(alpha=None, beta=None) -> QuantumCircuit:
    if alpha is None:
        alpha = 2 / np.sqrt(5)
    if beta is None:
        beta = 1 / np.sqrt(5)

    norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
    alpha, beta = alpha / norm, beta / norm

    qr = QuantumRegister(3, "q")
    cr = ClassicalRegister(2, "c")
    qc = QuantumCircuit(qr, cr)

    theta = 2 * np.arccos(np.clip(float(np.real(alpha)), -1, 1))
    qc.ry(theta, qr[0])
    qc.barrier()

    qc.h(qr[1])
    qc.cx(qr[1], qr[2])
    qc.barrier()

    qc.cx(qr[0], qr[1])
    qc.h(qr[0])
    qc.barrier()
    qc.measure(qr[0], cr[0])
    qc.measure(qr[1], cr[1])

    qc.x(qr[2]).c_if(cr[1], 1)
    qc.z(qr[2]).c_if(cr[0], 1)

    return qc


# Q2.2 - Long-distance CNOT via SWAP chain
def build_long_distance_cnot(control: int = 0, target: int = 4,
                              n_qubits: int = 5) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    steps = list(range(target - 1, control, -1))

    qc.barrier()
    for i in steps:
        qc.swap(i, i + 1)

    qc.barrier()
    qc.cx(control, control + 1)

    qc.barrier()
    for i in reversed(steps):
        qc.swap(i, i + 1)

    return qc
