# quantum_solver.py
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from typing import Tuple, List

# Pauli matrices (for exact Hamiltonian building)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)

def tfim_hamiltonian(n_qubits=6, J=1.0, h=1.0, periodic=False):
    N = n_qubits
    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)

    for i in range(N):
        j = (i + 1) % N if periodic else i + 1
        if j >= N:
            continue
        op = 1
        for q in range(N):
            if q == i or q == j:
                op = np.kron(op, Z)
            else:
                op = np.kron(op, I)
        H += -J * op

    for i in range(N):
        op = 1
        for q in range(N):
            if q == i:
                op = np.kron(op, X)
            else:
                op = np.kron(op, I)
        H += -h * op

    return H

# -------------------------
# Ansatze
# -------------------------
def ansatz_ry_chain(params: np.ndarray, n_qubits=6, depth=1) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    assert len(params) == depth * n_qubits
    idx = 0
    for _ in range(depth):
        for q in range(n_qubits):
            qc.ry(params[idx], q); idx += 1
        for q in range(n_qubits - 1):
            qc.cx(q, q+1)
    return qc

def ansatz_heavy_hex(params: np.ndarray, n_qubits=6, depth=1) -> QuantumCircuit:
    # Hardware-efficient ansatz (HEA-like): RY+RZ then entangle with CZ chain
    qc = QuantumCircuit(n_qubits)
    assert len(params) == depth * n_qubits * 2
    idx = 0
    for _ in range(depth):
        for q in range(n_qubits):
            qc.ry(params[idx], q); idx += 1
            qc.rz(params[idx], q); idx += 1
        # entangler: alternating CZ chain
        for q in range(n_qubits - 1):
            qc.cz(q, q+1)
    return qc

# -------------------------
# Statevector and sampling helpers
# -------------------------
def statevector_from_circuit(qc: QuantumCircuit) -> np.ndarray:
    sv = Statevector.from_instruction(qc)
    return np.array(sv.data, dtype=complex)

def sample_measurements_from_statevector(statevec: np.ndarray, shots: int,
                                         noise_prob: float = 0.0) -> Tuple[dict, List[str]]:
    """
    Emulate projective measurements (in Z-basis) by sampling from the statevector probabilities.
    Optionally apply simple bit-flip noise (flip each measured bit with prob=noise_prob).
    Returns counts dict and list of bitstrings (most recent first)
    """
    probs = np.abs(statevec)**2
    dim = len(probs)
    # sample indices according to probs
    choices = np.random.choice(dim, size=shots, p=probs)
    # convert to bitstrings MSB..LSB (we'll use qiskit's ordering convention)
    n_qubits = int(np.round(np.log2(dim)))
    bitstrings = []
    counts = {}
    for idx in choices:
        b = format(idx, '0{}b'.format(n_qubits))
        # apply independent bit-flip noise per qubit (simple model)
        if noise_prob > 0:
            b_list = list(b)
            for i in range(n_qubits):
                if np.random.rand() < noise_prob:
                    b_list[i] = '1' if b_list[i] == '0' else '0'
            b = ''.join(b_list)
        counts[b] = counts.get(b, 0) + 1
        bitstrings.append(b)
    return counts, bitstrings

def expectation_from_counts_ZZ(counts: dict, n_qubits: int, J: float) -> float:
    """
    Compute expectation of -J * sum Z_i Z_{i+1} from Z-basis measurement counts.
    """
    total_shots = sum(counts.values()) if counts else 1
    energy = 0.0
    for bitstr, c in counts.items():
        # map '0'->+1, '1'->-1
        spins = [1 if ch == '0' else -1 for ch in bitstr]
        zz_sum = 0
        for i in range(n_qubits - 1):
            zz_sum += spins[i] * spins[i+1]
        energy += (-J) * zz_sum * (c / total_shots)
    return energy

def expectation_from_counts_X(counts: dict, n_qubits: int, h: float) -> float:
    """
    For X expectation, we can measure in X-basis by applying H to the state first.
    But here counts are assumed to be from sampling in X-basis already.
    For single-bitstrings measured in X-basis, map '0'->+1, '1'->-1 per qubit and sum.
    """
    total_shots = sum(counts.values()) if counts else 1
    energy = 0.0
    for bitstr, c in counts.items():
        spins = [1 if ch == '0' else -1 for ch in bitstr]
        energy += (-h) * sum(spins) * (c / total_shots)
    return energy

def estimate_energy_from_statevector(statevec: np.ndarray, n_qubits: int, J: float, h: float,
                                      shots: int = 0, noise_prob: float = 0.0,
                                      repeats: int = 1) -> float:
    """
    Improved shot estimator:
    - shots==0 -> exact expectation via matrix math
    - shots>0  -> split shots between Z and X measurements and average over `repeats`
    """
    if shots <= 0:
        N = n_qubits
        H = tfim_hamiltonian(N, J, h)
        return float(np.real(np.conjugate(statevec) @ (H @ statevec)))

    shots_z = max(1, shots // 2)
    shots_x = max(1, shots - shots_z)

    total = 0.0
    for _ in range(max(1, repeats)):
        counts_z, _ = sample_measurements_from_statevector(statevec, shots_z, noise_prob=noise_prob)
        zz_energy = expectation_from_counts_ZZ(counts_z, n_qubits, J)

        # X-basis: apply global Hadamard then sample
        H_single = np.array([[1,1],[1,-1]]) / np.sqrt(2)
        H_global = H_single
        for _ in range(n_qubits - 1):
            H_global = np.kron(H_global, H_single)
        statevec_x = H_global @ statevec
        counts_x, _ = sample_measurements_from_statevector(statevec_x, shots_x, noise_prob=noise_prob)
        x_energy = expectation_from_counts_X(counts_x, n_qubits, h)

        total += (zz_energy + x_energy)

    return float(total / max(1, repeats))


# -------------------------
# Optimization wrapper
# -------------------------
_global_trace = []

def traceable_energy(params, n_qubits, depth, J, h, shots, noise_prob, ansatz_type, repeats):
    if ansatz_type == 'ry':
        qc = ansatz_ry_chain(params, n_qubits=n_qubits, depth=depth)
    else:
        qc = ansatz_heavy_hex(params, n_qubits=n_qubits, depth=depth)
    statevec = statevector_from_circuit(qc)
    val = estimate_energy_from_statevector(statevec, n_qubits, J, h, shots=shots, noise_prob=noise_prob, repeats=repeats)
    _global_trace.append(val)
    return val

def run_vqe(n_qubits=6, depth=1, J=1.0, h=1.0,
            restarts=3, maxiter=400, method='COBYLA',
            ansatz_type='ry', shots=0, noise_prob=0.0,
            repeats=1, compare_depths: List[int] = None):
    """
    Added 'method' (optimizer method) and 'repeats' (averaging repeats for shot mode).
    """
    if compare_depths:
        results = {}
        for d in compare_depths:
            results[d] = run_vqe(n_qubits=n_qubits, depth=d, J=J, h=h,
                                 restarts=restarts, maxiter=maxiter, method=method,
                                 ansatz_type=ansatz_type, shots=shots, noise_prob=noise_prob,
                                 repeats=repeats, compare_depths=None)
        return {"compare_depths": results}

    param_count = (depth * n_qubits) if ansatz_type == 'ry' else (depth * n_qubits * 2)
    best = {"energy": float("inf"), "params": None, "trace": None, "success": False, "message": ""}

    for r in range(restarts):
        x0 = np.random.uniform(0, 2*np.pi, param_count)
        global _global_trace
        _global_trace = []
        try:
            res = minimize(lambda p: traceable_energy(p, n_qubits, depth, J, h, shots, noise_prob, ansatz_type, repeats),
                           x0=x0, method=method, options={'maxiter': maxiter, 'disp': False})
        except Exception:
            res = minimize(lambda p: traceable_energy(p, n_qubits, depth, J, h, shots, noise_prob, ansatz_type, repeats),
                           x0=x0, method='Nelder-Mead', options={'maxiter': maxiter, 'disp': False})

        if res.fun < best["energy"]:
            best["energy"] = float(res.fun)
            best["params"] = res.x.tolist()
            best["trace"] = _global_trace.copy()
            best["success"] = bool(res.success)
            best["message"] = getattr(res, 'message', '')

    return best

def exact_ground_energy(n_qubits=6, J=1.0, h=1.0):
    H = tfim_hamiltonian(n_qubits=n_qubits, J=J, h=h)
    vals, _ = np.linalg.eigh(H)
    return float(np.min(vals))
