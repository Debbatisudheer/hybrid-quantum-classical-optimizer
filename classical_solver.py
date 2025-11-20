# classical_solver.py
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from quantum_solver import tfim_hamiltonian, exact_ground_energy, expectation_from_counts_ZZ, expectation_from_counts_X, ansatz_ry_chain, ansatz_heavy_hex, estimate_energy_from_statevector

# We'll reuse ansatz builders from quantum_solver if needed; simple wrappers here

def build_ansatz(params, n_qubits=6, depth=1, ansatz_type='ry'):
    if ansatz_type == 'ry':
        return ansatz_ry_chain(params, n_qubits=n_qubits, depth=depth)
    else:
        return ansatz_heavy_hex(params, n_qubits=n_qubits, depth=depth)

def classical_obj_statevector(params, n_qubits, depth, J, h, ansatz_type):
    qc = build_ansatz(params, n_qubits=n_qubits, depth=depth, ansatz_type=ansatz_type)
    sv = Statevector.from_instruction(qc)
    H = tfim_hamiltonian(n_qubits=n_qubits, J=J, h=h)
    return float(np.real(np.conjugate(sv.data) @ (H @ sv.data)))

def classical_optimize_same_ansatz(n_qubits=6, depth=1, restarts=3, maxiter=400, method='BFGS', ansatz_type='ry', shots=0, noise_prob=0.0, repeats=1):
    best = {"energy": float("inf"), "params": None, "trace": None}
    param_count = (depth * n_qubits) if ansatz_type == 'ry' else (depth * n_qubits * 2)

    for _ in range(restarts):
        x0 = np.random.uniform(0, 2*np.pi, param_count)
        trace = []

        def obj(p):
            qc = build_ansatz(p, n_qubits=n_qubits, depth=depth, ansatz_type=ansatz_type)
            sv = Statevector.from_instruction(qc).data
            val = estimate_energy_from_statevector(sv, n_qubits=n_qubits, J=1.0, h=1.0, shots=shots, noise_prob=noise_prob, repeats=repeats)
            trace.append(val)
            return val

        try:
            res = minimize(obj, x0=x0, method=method, options={'maxiter': maxiter, 'disp': False})
        except Exception:
            res = minimize(obj, x0=x0, method='Nelder-Mead', options={'maxiter': maxiter, 'disp': False})

        if res.fun < best["energy"]:
            best["energy"] = float(res.fun)
            best["params"] = res.x.tolist()
            best["trace"] = trace.copy()

    return best

