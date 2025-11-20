# app.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from quantum_solver import run_vqe, exact_ground_energy
from classical_solver import classical_optimize_same_ansatz

# --------------------------------------------------------
# PAGE SETUP
# --------------------------------------------------------
st.set_page_config(page_title="Quantum–Classical Hybrid VQE Optimizer", layout="wide")
st.title("🔬 Quantum–Classical Hybrid VQE Optimizer")

# --------------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------------
with st.sidebar:
    st.header("Run Controls")

    # Preset for accuracy mode
    if st.button("Preset: Accuracy Mode"):
        st.session_state['preset_msg'] = "Use: Depth=2, Optimizer=Nelder-Mead, Restarts=4, Maxiter=1200, Mode=statevector"


    if 'preset_msg' in st.session_state:
        st.caption(st.session_state['preset_msg'])

    # Basic inputs
    n_qubits = st.number_input("Number of Qubits", min_value=4, max_value=8, value=6, step=1)
    depth = st.number_input("Ansatz Depth (layers)", min_value=1, max_value=4, value=1, step=1)

    ansatz_type = st.selectbox("Ansatz Type", options=['ry', 'hea (HEA)'], index=0)
    ansatz_key = 'ry' if ansatz_type.startswith('ry') else 'hea'

    mode = st.selectbox("Simulation Mode", options=['statevector (exact)', 'shots (emulated)'], index=0)

    # Shot-mode settings
    if mode == "shots (emulated)":
        shots = st.number_input("Shots (for emulated mode)", min_value=200, max_value=10000, value=2000, step=200)
        noise = st.slider("Measurement noise (0=no noise)", 0.0, 0.5, 0.00, step=0.01)
        repeats = st.number_input("Averaging repeats", min_value=1, max_value=10, value=2, step=1)
    else:
        shots = 0
        noise = 0.0
        repeats = 1
        st.info("Shots & noise disabled in exact mode.")

    # Hamiltonian
    J = st.number_input("Coupling J", value=1.0, step=0.1)
    h = st.number_input("Transverse Field h", value=1.0, step=0.1)

    # Optimizer
    optimizer = st.selectbox("Optimizer Method", options=['COBYLA', 'Nelder-Mead', 'BFGS', 'SLSQP'], index=0)
    restarts = st.slider("Restarts", 1, 8, 3)
    maxiter = st.number_input("Max iterations per run", min_value=50, max_value=5000, value=800, step=50)

    # Auto-tune mode
    auto_tune = st.checkbox("✨ Auto-Tune Best (Depth + Optimizer)", value=False)

    run_button = st.button("Run full comparison")

# --------------------------------------------------------
# DESCRIPTION
# --------------------------------------------------------
st.markdown("""
### 🤖 What this demo does:

📱 **Mobile-Friendly App**

This project runs all quantum & classical computations on the server,
so it works smoothly even on mobile devices.

If a run takes a few seconds, please wait — the server is processing
the optimization.

Goal: We check how close quantum computing comes to classical computing when solving the same optimization problem

## 🧭 Start Here — Recommended Order

Before changing any settings, please try these two steps:

### 1️⃣ Pre-Test (Quick Check)
Run the **Pre-Test** to verify that:
- Quantum circuit is working  
- Hamiltonian is correct  
- Simulator is running  
This helps you confirm everything is OK before doing full comparisons.

### 2️⃣ Auto-Tune (Automatically picks best settings)
Auto-Tune will choose recommended values for:
- Depth  
- Optimizer  
- Max iterations  
- Restarts  

This gives you a good starting point without confusion.

---

### After Pre-Test + Auto-Tune:
You can now run the **Full Quantum–Classical Comparison** with:
- Exact simulation  
- Shot-based simulation  
- Noise  
- Different depths  
- Different ansatz types  

This shows how **Quantum VQE** and **Classical optimization** perform on the same problem.

It performs three steps:

**1️⃣ Exact Calculation**  
Gets the true lowest energy using normal mathematical diagonalization (baseline answer).

**2️⃣ Quantum VQE**  
Uses a parameterized quantum circuit (ansatz) and an optimizer to find the minimum energy.

**3️⃣ Classical Optimization**  
Runs a classical optimizer on the same ansatz to compare performance.

### 🎛 What you can test:

- **Exact simulation** (no noise, perfect accuracy)  
- **Shot-based simulation** (like real quantum hardware)  
- **Noise simulation** (to see effect of measurement errors)  
- **Different ansatz depths** (circuit complexity)  
- **Hardware-efficient ansatz (HEA)**  

This helps you understand how classical and quantum methods behave on the same task.

---
👨‍💻 **Developed by Sudheer Debbati**  
🔗 *Quantum–Classical Hybrid VQE Research Project*

""")

# --------------------------------------------------------
# RUN COMPUTATION
# --------------------------------------------------------
if run_button:

    # Compute exact energy
    with st.spinner("Computing exact ground energy..."):
        exact = exact_ground_energy(n_qubits=n_qubits, J=J, h=h)

    st.subheader("Exact Ground Energy")
    st.write(exact)

    # --------------------------------------------------------
    # AUTO-TUNE MODE
    # --------------------------------------------------------
    if auto_tune:
        st.subheader("✨ Auto-Tune: Finding Best Depth & Optimizer")

        depths = [1, 2, 3]
        optimizers = ['COBYLA', 'Nelder-Mead', 'BFGS']

        best_energy = 999
        best_config = None
        table = []

        total_runs = len(depths) * len(optimizers)
        progress = st.progress(0)
        count = 0

        for d in depths:
            for opt in optimizers:
                count += 1
                progress.progress(count / total_runs)

                with st.spinner(f"Running depth={d}, optimizer={opt}..."):
                    qres = run_vqe(
                        n_qubits=n_qubits,
                        depth=d,
                        J=J, h=h,
                        restarts=restarts,
                        maxiter=maxiter,
                        ansatz_type=ansatz_key,
                        shots=shots,
                        noise_prob=noise,
                        repeats=repeats,
                        method=opt
                    )

                e = qres["energy"]
                table.append((d, opt, e))

                if e < best_energy:
                    best_energy = e
                    best_config = (d, opt, qres)

        st.success("Auto tuning complete!")

        bd, bo, bres = best_config

        st.subheader("🌟 Best Configuration Found")
        st.write(f"**Best Depth:** {bd}")
        st.write(f"**Best Optimizer:** {bo}")
        st.write(f"**Best Energy:** {best_energy}")

        st.json(bres)

        st.subheader("📊 Full Auto-Tune Table")
        for d, opt, e in table:
            st.write(f"Depth {d}, Optimizer {opt} → Energy = {e}")

        # Plot best trace
        trace = bres.get("trace", [])

        fig, ax = plt.subplots(figsize=(8, 4))
        if trace:
            ax.plot(trace, label=f"Best: depth={bd}, opt={bo}")

        ax.axhline(exact, linestyle="--", color="black", label="Exact")
        ax.set_xlabel("Evaluations")
        ax.set_ylabel("Energy")
        ax.legend()
        st.pyplot(fig)

        st.stop()

    # --------------------------------------------------------
    # MANUAL MODE
    # --------------------------------------------------------

    # Quantum VQE
    with st.spinner("Running Quantum VQE..."):
        qres = run_vqe(
            n_qubits=n_qubits, depth=depth,
            J=J, h=h,
            restarts=restarts,
            maxiter=maxiter,
            ansatz_type=ansatz_key,
            shots=shots,
            noise_prob=noise,
            repeats=repeats,
            method=optimizer
        )

    st.subheader("Quantum VQE Result")
    st.json(qres)

    # Classical solver
    with st.spinner("Running Classical Optimizer..."):
        cres = classical_optimize_same_ansatz(
            n_qubits=n_qubits,
            depth=depth,
            restarts=restarts,
            maxiter=maxiter,
            ansatz_type=ansatz_key,
            shots=shots,
            noise_prob=noise,
            repeats=repeats,
            method=optimizer
        )

    st.subheader("Classical Optimizer Result")
    st.json(cres)

    # Plot traces
    fig, ax = plt.subplots(figsize=(8, 4))

    def flat(t):
        if not t: return []
        out = []
        for x in t:
            if isinstance(x, (int, float)):
                out.append(x)
            elif isinstance(x, list):
                out.extend(x)
        return out

    qtrace = flat(qres.get("trace", []))
    ctrace = flat(cres.get("trace", []))

    if qtrace:
        ax.plot(qtrace, label="Quantum (VQE)")
    if ctrace:
        ax.plot(ctrace, label="Classical")

    ax.axhline(exact, linestyle="--", color="black", label="Exact")
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Energy")
    ax.legend()
    st.pyplot(fig)

    st.success("Finished!")

else:
    st.info("Configure parameters and click **Run full comparison**.")
