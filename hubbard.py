"""
hubbard.py  -  Phase 3: Fermi-Hubbard Model via Trotterization
================================================================
4-qubit encoding of a 2-site dimer:
  q0 : Site 1, Spin Up   (↑)
  q1 : Site 1, Spin Down (↓)
  q2 : Site 2, Spin Up   (↑)
  q3 : Site 2, Spin Down (↓)

Jordan-Wigner mapping (from Appendix):
  Hopping term  : H_J = (J/2)(X_j X_{j+1} + Y_j Y_{j+1})  → RXX + RYY gates
  Interaction   : H_U = (U/4)(I - Z_{j↑} - Z_{j↓} + Z_{j↑}Z_{j↓})  → RZ + RZZ gates

One Trotter step for time slice dt:
  U(dt) ≈ exp(-i H_J^{↑} dt) · exp(-i H_J^{↓} dt) · exp(-i H_U^{site1} dt) · exp(-i H_U^{site2} dt)
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

simulator = AerSimulator()


# ─────────────────────────────────────────────────────────────────────────────
#  Primitive gate blocks  (all derived from Jordan-Wigner in the Appendix)
# ─────────────────────────────────────────────────────────────────────────────

def _rxx(qc: QuantumCircuit, theta: float, q0: int, q1: int):
    """
    RXX(theta) = exp(-i theta/2 * X⊗X)
    Implements the XX part of the hopping term.
    Decomposition: CNOT(q0,q1) · Rx(theta, q0) · CNOT(q0,q1)
    """
    qc.cx(q0, q1)
    qc.rx(theta, q0)
    qc.cx(q0, q1)


def _ryy(qc: QuantumCircuit, theta: float, q0: int, q1: int):
    """
    RYY(theta) = exp(-i theta/2 * Y⊗Y)
    Implements the YY part of the hopping term.
    Decomposition: S(q0)·S(q1) · CNOT(q0,q1) · Rx(theta, q0) · CNOT(q0,q1) · Sdg(q0)·Sdg(q1)
    """
    qc.s(q0);  qc.s(q1)
    qc.cx(q0, q1)
    qc.rx(theta, q0)
    qc.cx(q0, q1)
    qc.sdg(q0); qc.sdg(q1)


def _rzz(qc: QuantumCircuit, theta: float, q0: int, q1: int):
    """
    RZZ(theta) = exp(-i theta/2 * Z⊗Z)
    Implements the ZZ interaction term (entangling part of H_U).
    Decomposition: CNOT · Rz(theta) · CNOT
    """
    qc.cx(q0, q1)
    qc.rz(theta, q1)
    qc.cx(q0, q1)


def _hopping_block(qc: QuantumCircuit, J: float, dt: float,
                   q_site1: int, q_site2: int, label: str = ""):
    """
    exp(-i H_J dt) for one spin channel between site1 and site2.
    H_J = (J/2)(XX + YY)
    => exp(-i (J/2) dt * XX) · exp(-i (J/2) dt * YY)

    The angle passed to RXX/RYY is 2*(J/2)*dt = J*dt
    because RXX(theta) = exp(-i theta/2 XX), so theta = J*dt.
    """
    angle = J * dt
    if label:
        qc.barrier(label=label)
    _rxx(qc, angle, q_site1, q_site2)
    _ryy(qc, angle, q_site1, q_site2)


def _interaction_block(qc: QuantumCircuit, U: float, dt: float,
                       q_up: int, q_down: int, label: str = ""):
    """
    exp(-i H_U dt) for one site with spin-up qubit q_up and spin-down qubit q_down.

    H_U = (U/4)(I - Z_{up} - Z_{down} + Z_{up}Z_{down})

    The I term gives a global phase (ignored).
    -Z_{up}  → Rz(+U/2 * dt)   [Rz(phi) = exp(-i phi/2 Z), so phi = U*dt/2... 
                                  but H has coefficient -U/4, evolution gives
                                  exp(-i*(-U/4)*dt * Z) = exp(+i U dt/4 * Z)
                                  = Rz(-U*dt/2) since Rz(phi)=exp(-i phi/2 Z)]
    We work out each term carefully:
      exp(-i * (-U/4) * dt * Z_up)   = Rz(phi) with phi/2 = -(-U/4)*dt  => phi = U*dt/2  => Rz(U*dt/2)
      exp(-i * (-U/4) * dt * Z_down) = Rz(U*dt/2) on q_down
      exp(-i * (U/4)  * dt * ZZ)     = RZZ(phi) with phi = U*dt/2
    """
    phi = U * dt / 2
    if label:
        qc.barrier(label=label)
    qc.rz(phi, q_up)
    qc.rz(phi, q_down)
    _rzz(qc, phi, q_up, q_down)


# ─────────────────────────────────────────────────────────────────────────────
#  Single Trotter step  (Q3.1)
# ─────────────────────────────────────────────────────────────────────────────

def build_trotter_step(J: float = 1.0, U: float = 0.0, dt: float = 0.1) -> QuantumCircuit:
    """
    One Trotter step for the 2-site Fermi-Hubbard dimer.
    Qubit layout:
      q0 = site1-up, q1 = site1-down, q2 = site2-up, q3 = site2-down

    Gate sequence (labelled for Q3.1):
      1. Hopping ↑ channel  : q0 <-> q2  (spin-up hopping, includes Z-string via RXX+RYY)
      2. Hopping ↓ channel  : q1 <-> q3  (spin-down hopping)
      3. Interaction site 1 : q0, q1     (on-site Coulomb repulsion)
      4. Interaction site 2 : q2, q3
    """
    qc = QuantumCircuit(4, name=f"Trotter(J={J},U={U},dt={dt:.3f})")

    # ── Hopping terms (Jordan-Wigner → XY rotation = RXX + RYY) ─────────────
    # Spin-up channel: site1(q0) <-> site2(q2)
    # Note: q0 and q2 are next-nearest neighbors in qubit ordering.
    # The Z-string from JW transformation for the hop q0->q2 picks up a Z_q1 factor.
    # For nearest-neighbor hops the strings cancel; here we include it explicitly.
    qc.barrier(label="Hop ↑: q0↔q2 (Z-string on q1)")
    # Z-string contribution: conjugate q1 around the RXX/RYY
    qc.cz(0, 1)               # start Z-string
    _rxx(qc, J * dt, 0, 2)
    _ryy(qc, J * dt, 0, 2)
    qc.cz(0, 1)               # end Z-string (CZ is its own inverse)

    # Spin-down channel: site1(q1) <-> site2(q3)  (nearest neighbor in spin-down subspace)
    qc.barrier(label="Hop ↓: q1↔q3 (Z-string on q2)")
    qc.cz(1, 2)
    _rxx(qc, J * dt, 1, 3)
    _ryy(qc, J * dt, 1, 3)
    qc.cz(1, 2)

    # ── Interaction terms (Jordan-Wigner → RZ + RZZ) ────────────────────────
    _interaction_block(qc, U, dt, q_up=0, q_down=1, label="Interaction site 1")
    _interaction_block(qc, U, dt, q_up=2, q_down=3, label="Interaction site 2")

    return qc


# ─────────────────────────────────────────────────────────────────────────────
#  Time evolution  (Q3.2 and Q3.3)
# ─────────────────────────────────────────────────────────────────────────────

def evolve(initial_state: str, J: float, U: float,
           tau_max: float = np.pi, n_time_points: int = 50,
           n_trotter_steps: int = 20) -> tuple:
    """
    Simulate time evolution of the 4-qubit Fermi-Hubbard system.

    Parameters
    ----------
    initial_state : str
        4-bit string, e.g. '1000' means q0=1, q1=0, q2=0, q3=0
        (Qiskit convention: rightmost bit = q0)
    J, U         : Hamiltonian parameters
    tau_max      : total evolution time
    n_time_points: number of time snapshots
    n_trotter_steps: Trotter steps per time point (finer = more accurate)

    Returns
    -------
    times  : np.ndarray of shape (n_time_points,)
    probs  : dict mapping state_label -> np.ndarray of probabilities
    """
    times = np.linspace(0, tau_max, n_time_points)
    # We track all basis states but return a dict for the ones the caller cares about
    all_probs = {format(i, "04b"): np.zeros(n_time_points) for i in range(16)}

    for t_idx, tau in enumerate(times):
        if tau == 0:
            # At t=0, probability is 1 for initial state
            all_probs[initial_state][t_idx] = 1.0
            continue

        dt = tau / n_trotter_steps

        # Build circuit: prepare initial state + n_trotter_steps Trotter steps
        qc = QuantumCircuit(4)

        # Prepare initial state
        # initial_state string: '1000' means bit3=1,bit2=0,bit1=0,bit0=0
        # Qiskit qubit ordering: q0=rightmost bit
        for qubit_idx, bit in enumerate(reversed(initial_state)):
            if bit == "1":
                qc.x(qubit_idx)

        # Append Trotter steps
        step = build_trotter_step(J=J, U=U, dt=dt)
        for _ in range(n_trotter_steps):
            qc = qc.compose(step)

        # Get statevector
        qc.save_statevector()
        job = simulator.run(qc)
        sv = np.array(job.result().get_statevector())

        # Compute probabilities for each basis state
        for i, amp in enumerate(sv):
            label = format(i, "04b")
            all_probs[label][t_idx] = abs(amp) ** 2

    return times, all_probs


# ─────────────────────────────────────────────────────────────────────────────
#  Analytical reference curves
# ─────────────────────────────────────────────────────────────────────────────

def rabi_transfer(J: float, times: np.ndarray) -> np.ndarray:
    """
    Analytical probability of transfer |1000> -> |0010> for U=0.
    Two-level Rabi formula: P(t) = sin^2(J * t)
    (The hopping couples |1000> and |0010> with matrix element J,
     giving Rabi oscillation at frequency J.)
    """
    return np.sin(J * times) ** 2
