# streamlit_asteroid_gui.py
# Streamlit GUI: ML-Accelerated Asteroid Trajectory Simulation (GNN + LSTM Surrogate)
#
# Install (local):
#   pip install streamlit numpy torch matplotlib
#
# Run:
#   streamlit run streamlit_asteroid_gui.py
#
# Streamlit Cloud tips:
# - Use a headless Matplotlib backend ("Agg") for cloud.
# - Pin versions in requirements.txt + runtime.txt (python-3.11).
#
# Improvements included:
# ✅ Headless-safe matplotlib backend (Agg)
# ✅ Consistent Sun-fixed physics (Sun immovable AND not accelerated)
# ✅ Vectorized gravity for faster physics steps
# ✅ ML normalization for asteroid AND planet positions (more stable training)
# ✅ ML rollout uses true future planet positions (more realistic + lower error)
# ✅ Reset model button + safer caching behavior
# ✅ Robust training loop: checks NaNs/Infs, stronger grad clipping, shows full traceback
# ✅ Cloud-safe defaults: smaller batches, reduced threads
# ✅ Extra metrics: final-step error, relative % error

import time
import math
import traceback
import numpy as np
import streamlit as st

# --- Headless-safe Matplotlib import (Streamlit Cloud friendly) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Torch ---
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Constants
# -----------------------------
G = 4 * math.pi**2  # AU^3 / (yr^2 * solar_mass)
AU_TO_KM = 149_597_870.7

PLANETS = [
    ("Sun",   1.0,      0.0,    0.0),
    ("Venus", 2.447e-6, 0.723,  0.615),
    ("Earth", 3.003e-6, 1.000,  1.000),
    ("Mars",  3.213e-7, 1.524,  1.881),
]

DIM = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Normalization scales (keep consistent with UI bounds)
AST_SCALE_POS = 2.5    # AU (fits slider max ~2.5)
AST_SCALE_VEL = 10.0   # AU/yr (safe upper bound in this toy system)


# -----------------------------
# Physics helpers (Sun-fixed, vectorized)
# -----------------------------
def circular_orbit_state(a_au, period_yr, phase):
    w = 2 * math.pi / period_yr
    x = a_au * math.cos(phase)
    y = a_au * math.sin(phase)
    vx = -a_au * w * math.sin(phase)
    vy =  a_au * w * math.cos(phase)
    return np.array([x, y], dtype=np.float64), np.array([vx, vy], dtype=np.float64)

def accel_all_vectorized_sun_fixed(pos, masses):
    """
    Vectorized acceleration, consistent Sun-fixed frame.
    Sun is pinned at origin and has zero acceleration.
    """
    # r_ij = x_j - x_i
    r = pos[None, :, :] - pos[:, None, :]         # (N,N,2)
    dist2 = (r**2).sum(axis=-1) + 1e-12           # (N,N)
    inv_dist3 = 1.0 / (dist2 * np.sqrt(dist2))    # (N,N)
    np.fill_diagonal(inv_dist3, 0.0)

    a_pair = G * (r * inv_dist3[..., None]) * masses[None, :, None]  # (N,N,2)
    acc = a_pair.sum(axis=1)  # (N,2)

    acc[0] = 0.0  # Sun fixed
    return acc

def leapfrog_step(pos, vel, masses, dt):
    a0 = accel_all_vectorized_sun_fixed(pos, masses)
    vel_half = vel + 0.5 * dt * a0
    pos_new = pos + dt * vel_half
    a1 = accel_all_vectorized_sun_fixed(pos_new, masses)
    vel_new = vel_half + 0.5 * dt * a1

    # Pin Sun exactly
    pos_new[0] = 0.0
    vel_new[0] = 0.0
    return pos_new, vel_new

def simulate_system(
    steps=365,
    dt=1/365,
    seed=0,
    asteroid_init=None,  # dict with r0, v0 (AU, AU/yr)
    asteroid_mass=1e-15
):
    rng = np.random.default_rng(seed)

    masses = np.array([p[1] for p in PLANETS] + [asteroid_mass], dtype=np.float64)
    N = len(PLANETS) + 1  # planets + asteroid

    pos = np.zeros((N, DIM), dtype=np.float64)
    vel = np.zeros((N, DIM), dtype=np.float64)

    # Sun fixed
    pos[0] = np.array([0.0, 0.0])
    vel[0] = np.array([0.0, 0.0])

    # Planets: circular orbits with random phases
    for i, (_, _, a, T) in enumerate(PLANETS[1:], start=1):
        phase = rng.uniform(0, 2*np.pi)
        r, v = circular_orbit_state(a, T, phase)
        pos[i], vel[i] = r, v

    # Asteroid init
    if asteroid_init is None:
        a_ast = rng.uniform(0.7, 2.2)
        phase = rng.uniform(0, 2*np.pi)
        r0 = np.array([a_ast * math.cos(phase), a_ast * math.sin(phase)], dtype=np.float64)
        r = np.linalg.norm(r0) + 1e-12
        v_circ = math.sqrt(G * masses[0] / r)
        tdir = np.array([-r0[1], r0[0]]) / r
        rdir = r0 / r
        speed_factor = rng.uniform(0.85, 1.15)
        radial_factor = rng.uniform(-0.08, 0.08)
        v0 = speed_factor * v_circ * tdir + radial_factor * v_circ * rdir
    else:
        r0 = np.array(asteroid_init["r0"], dtype=np.float64)
        v0 = np.array(asteroid_init["v0"], dtype=np.float64)

    pos[-1], vel[-1] = r0, v0

    planet_pos = np.zeros((steps, len(PLANETS), DIM), dtype=np.float64)
    asteroid_state = np.zeros((steps, 4), dtype=np.float64)

    for t in range(steps):
        planet_pos[t] = pos[:len(PLANETS)]
        asteroid_state[t] = np.array([pos[-1,0], pos[-1,1], vel[-1,0], vel[-1,1]])
        pos, vel = leapfrog_step(pos, vel, masses, dt)

    return planet_pos, asteroid_state


# -----------------------------
# ML Dataset (normalizes planets + asteroid)
# -----------------------------
class TrajDataset(Dataset):
    def __init__(self, planet_pos, ast_state, past_len=30, future_len=30):
        # planet_pos: (Ntraj, steps, P, 2)
        # ast_state : (Ntraj, steps, 4)
        self.planet_pos = planet_pos
        self.ast_state = ast_state
        self.past_len = past_len
        self.future_len = future_len
        self.Ntraj, self.steps = ast_state.shape[0], ast_state.shape[1]
        self.max_start = self.steps - (past_len + future_len)
        if self.max_start < 0:
            raise ValueError("Not enough steps for given past_len + future_len")

    def __len__(self):
        return self.Ntraj * (self.max_start + 1)

    def __getitem__(self, idx):
        traj_id = idx // (self.max_start + 1)
        s = idx % (self.max_start + 1)

        past_planets = self.planet_pos[traj_id, s:s+self.past_len].astype(np.float32)  # (T,P,2)
        fut_planets  = self.planet_pos[traj_id, s+self.past_len:s+self.past_len+self.future_len].astype(np.float32)

        past_ast = self.ast_state[traj_id, s:s+self.past_len].astype(np.float32)  # (T,4)
        fut_ast  = self.ast_state[traj_id, s+self.past_len:s+self.past_len+self.future_len].astype(np.float32)

        # Normalize planets (positions in AU)
        past_planets /= AST_SCALE_POS
        fut_planets  /= AST_SCALE_POS

        # Normalize asteroid state
        past_ast[:, 0:2] /= AST_SCALE_POS
        past_ast[:, 2:4] /= AST_SCALE_VEL
        fut_ast[:, 0:2]  /= AST_SCALE_POS
        fut_ast[:, 2:4]  /= AST_SCALE_VEL

        return (
            torch.tensor(past_planets, dtype=torch.float32),
            torch.tensor(past_ast, dtype=torch.float32),
            torch.tensor(fut_planets, dtype=torch.float32),
            torch.tensor(fut_ast, dtype=torch.float32),
        )


# -----------------------------
# Hybrid Model: GNN + LSTM
# -----------------------------
class SimpleGNNBlock(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, planet_xy, planet_m, ast_xy):
        # planet_xy: (B,P,2), planet_m:(P,), ast_xy:(B,2)
        B, P, _ = planet_xy.shape
        ast_xy_e = ast_xy.unsqueeze(1).expand(B, P, 2)
        d = planet_xy - ast_xy_e
        dist2 = (d**2).sum(dim=-1) + 1e-6
        invdist = 1.0 / torch.sqrt(dist2)
        m = planet_m.view(1, P).expand(B, P)
        feats = torch.cat([d, invdist.unsqueeze(-1), m.unsqueeze(-1)], dim=-1)  # (B,P,4)
        msg = self.mlp(feats)  # (B,P,H)
        return msg.sum(dim=1)  # (B,H)

class AsteroidSurrogate(nn.Module):
    def __init__(self, gnn_hidden=64, lstm_hidden=128):
        super().__init__()
        self.gnn = SimpleGNNBlock(hidden=gnn_hidden)
        self.inp = nn.Sequential(
            nn.Linear(4 + gnn_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )
        self.register_buffer("planet_masses", torch.tensor([p[1] for p in PLANETS], dtype=torch.float32))

    def forward(self, past_planets_xy, past_ast_state, future_planets_xy=None, future_len=30):
        """
        Inputs are normalized.
        past_planets_xy: (B,T,P,2)
        past_ast_state : (B,T,4)
        future_planets_xy: (B,F,P,2) or None
        Returns normalized predicted asteroid states: (B,F,4)
        """
        B, T, P, _ = past_planets_xy.shape

        feats = []
        for t in range(T):
            planets_t = past_planets_xy[:, t]
            ast_xy_t = past_ast_state[:, t, 0:2]
            agg = self.gnn(planets_t, self.planet_masses, ast_xy_t)
            f = torch.cat([past_ast_state[:, t], agg], dim=-1)
            feats.append(self.inp(f).unsqueeze(1))

        x = torch.cat(feats, dim=1)          # (B,T,128)
        out, (h, c) = self.lstm(x)
        last = out[:, -1]
        cur_state = past_ast_state[:, -1]    # normalized

        preds = []
        planets_last = past_planets_xy[:, -1]  # fallback

        for k in range(future_len):
            delta = self.head(last)
            next_state = cur_state + delta
            preds.append(next_state.unsqueeze(1))

            planets_k = planets_last if future_planets_xy is None else future_planets_xy[:, k]
            agg = self.gnn(planets_k, self.planet_masses, next_state[:, 0:2])
            step_in = self.inp(torch.cat([next_state, agg], dim=-1)).unsqueeze(1)
            out, (h, c) = self.lstm(step_in, (h, c))
            last = out[:, -1]
            cur_state = next_state

        return torch.cat(preds, dim=1)


# -----------------------------
# Training helpers
# -----------------------------
@st.cache_resource
def make_model():
    # Cloud-safe: CPU only is common; CUDA if available
    model = AsteroidSurrogate(gnn_hidden=64, lstm_hidden=128).to(DEVICE)
    return model

def train_quick(model, planet_pos, ast_state, past_len, future_len, epochs, batch_size, lr):
    ds = TrajDataset(planet_pos, ast_state, past_len=past_len, future_len=future_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()

    model.train()
    history = []

    # Cloud CPUs can be cranky with many threads
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    for ep in range(1, epochs + 1):
        running, n = 0.0, 0
        t0 = time.time()

        for step, (past_planets, past_ast, fut_planets, fut_ast) in enumerate(dl, start=1):
            try:
                past_planets = past_planets.to(DEVICE)
                past_ast = past_ast.to(DEVICE)
                fut_planets = fut_planets.to(DEVICE)
                fut_ast = fut_ast.to(DEVICE)

                pred = model(past_planets, past_ast, future_planets_xy=fut_planets, future_len=future_len)
                loss = loss_fn(pred, fut_ast)

                # If loss is NaN/Inf, skip this batch
                if not torch.isfinite(loss):
                    st.warning(f"Skipping batch {step}: non-finite loss ({float(loss.detach().cpu()):.6g})")
                    opt.zero_grad(set_to_none=True)
                    continue

                opt.zero_grad(set_to_none=True)
                loss.backward()

                # Stronger clipping helps prevent backward explosions
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                opt.step()

                running += float(loss.item()) * past_planets.size(0)
                n += past_planets.size(0)

            except RuntimeError:
                st.error("Training crashed (likely during backward). Full traceback:")
                st.code(traceback.format_exc())
                raise

        ep_loss = running / max(1, n)
        history.append(ep_loss)
        st.write(f"Epoch {ep}/{epochs}  loss={ep_loss:.6f}  time={time.time()-t0:.2f}s")

    return history


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Asteroid Trajectory ML Simulator", layout="wide")

st.title("☄️ ML Simulation of Asteroid Trajectories (Science Fair Demo)")
st.caption(
    "This app compares a physics-based N-body simulation to a machine-learning surrogate (GNN + LSTM) that predicts asteroid motion faster."
)

colA, colB = st.columns([1, 1])

with colA:
    st.subheader("1) Configure Simulation")
    years = st.slider("Simulation duration (years)", 0.2, 5.0, 2.0, 0.1)
    dt_days = st.slider("Time step (days)", 0.5, 5.0, 1.0, 0.5)
    steps = int((years * 365) / dt_days)
    dt = (dt_days / 365.0)

    st.markdown("**Asteroid initial orbit controls**")
    r_au = st.slider("Start distance from Sun (AU)", 0.6, 2.5, 1.2, 0.01)
    angle = st.slider("Start angle (degrees)", 0, 360, 30, 1)
    speed_factor = st.slider("Speed factor vs circular orbit", 0.6, 1.4, 1.05, 0.01)
    radial_push = st.slider("Radial push (adds in/out velocity)", -0.2, 0.2, 0.02, 0.01)

    seed = st.number_input("Random seed (planet starting phases)", min_value=0, max_value=10_000_000, value=42, step=1)

    th = math.radians(angle)
    r0 = np.array([r_au * math.cos(th), r_au * math.sin(th)], dtype=np.float64)
    r = np.linalg.norm(r0) + 1e-12
    v_circ = math.sqrt(G * PLANETS[0][1] / r)
    tdir = np.array([-r0[1], r0[0]]) / r
    rdir = r0 / r
    v0 = (speed_factor * v_circ) * tdir + (radial_push * v_circ) * rdir
    asteroid_init = {"r0": r0, "v0": v0}

    st.write("Asteroid initial state (AU, AU/yr):")
    st.code(f"r0 = {r0}\nv0 = {v0}", language="python")

with colB:
    st.subheader("2) Train ML Surrogate (optional but recommended)")
    st.write("Generate synthetic trajectories, then train a model to imitate the physics simulator.")

    # Cloud-safe defaults
    n_traj = st.slider("Number of synthetic trajectories", 50, 800, 200, 50)
    train_steps = st.slider("Steps per training trajectory", 200, 1200, min(600, steps), 50)

    past_len = st.slider("Past window (timesteps)", 10, 80, 30, 5)
    future_len = st.slider("Predict ahead (timesteps)", 10, 120, 60, 5)

    epochs = st.slider("Training epochs", 1, 30, 6, 1)
    batch_size = st.selectbox("Batch size (Cloud-safe: 32 or 64)", [32, 64, 128, 256], index=0)
    lr = st.selectbox("Learning rate", [1e-3, 2e-3, 5e-4], index=1)

    if "train_data" not in st.session_state:
        st.session_state.train_data = None
    if "trained" not in st.session_state:
        st.session_state.trained = False

    if st.button("Reset Model Weights"):
        st.cache_resource.clear()
        st.session_state.trained = False
        st.success("Model reset. You can retrain now.")

    if st.button("Generate Training Data"):
        with st.spinner("Generating trajectories..."):
            rng = np.random.default_rng(123)
            planet_pos_all = []
            ast_state_all = []
            t0 = time.time()
            for _ in range(n_traj):
                s = int(rng.integers(0, 10_000_000))
                pp, aa = simulate_system(steps=train_steps, dt=dt, seed=s)
                planet_pos_all.append(pp)
                ast_state_all.append(aa)
            planet_pos_all = np.stack(planet_pos_all, axis=0)
            ast_state_all = np.stack(ast_state_all, axis=0)
            st.session_state.train_data = (planet_pos_all, ast_state_all)
            st.session_state.trained = False
            st.success(f"Done! Generated {n_traj} trajectories in {time.time()-t0:.2f}s")

    model = make_model()

    if st.button("Train / Retrain Model"):
        if st.session_state.train_data is None:
            st.error("Generate training data first.")
        else:
            planet_pos_all, ast_state_all = st.session_state.train_data
            with st.spinner("Training model..."):
                history = train_quick(
                    model=model,
                    planet_pos=planet_pos_all,
                    ast_state=ast_state_all,
                    past_len=past_len,
                    future_len=future_len,
                    epochs=epochs,
                    batch_size=int(batch_size),
                    lr=float(lr),
                )
                st.session_state.trained = True

            fig = plt.figure()
            plt.plot(history)
            plt.xlabel("Epoch")
            plt.ylabel("Loss (SmoothL1)")
            plt.title("Training Loss")
            st.pyplot(fig, clear_figure=True)

st.divider()
st.subheader("3) Run Demo: Physics vs ML")

run_cols = st.columns([1, 1, 1, 1])
with run_cols[0]:
    run_demo = st.button("Run Simulation + Prediction")
with run_cols[1]:
    show_error_km = st.checkbox("Show error in km", value=True)
with run_cols[2]:
    show_planets = st.checkbox("Plot planet orbits", value=True)
with run_cols[3]:
    st.write(f"Device: **{DEVICE}**")

if run_demo:
    with st.spinner("Running physics simulation (truth)..."):
        t0 = time.time()
        planet_pos_true, ast_true = simulate_system(
            steps=steps, dt=dt, seed=int(seed), asteroid_init=asteroid_init
        )
        t_phys = time.time() - t0

    if steps < (past_len + future_len):
        st.error("Increase simulation duration or reduce past/future lengths so steps >= past_len + future_len.")
        st.stop()

    s = steps - (past_len + future_len)

    past_plan = planet_pos_true[s:s+past_len].astype(np.float32)                 # (T,P,2)
    future_plan = planet_pos_true[s+past_len:s+past_len+future_len].astype(np.float32)  # (F,P,2)

    past_ast = ast_true[s:s+past_len].astype(np.float32)                         # (T,4)
    true_future = ast_true[s+past_len:s+past_len+future_len].astype(np.float32)  # (F,4)

    # Normalize inputs for model
    past_plan_n = past_plan / AST_SCALE_POS
    future_plan_n = future_plan / AST_SCALE_POS

    past_ast_n = past_ast.copy()
    past_ast_n[:, 0:2] /= AST_SCALE_POS
    past_ast_n[:, 2:4] /= AST_SCALE_VEL

    if not st.session_state.get("trained", False):
        st.warning("Model is not trained yet. The ML prediction may be poor. Train it first for better results.")

    model = make_model()
    model.eval()

    x_plan = torch.tensor(past_plan_n, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    x_ast = torch.tensor(past_ast_n, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    x_future_plan = torch.tensor(future_plan_n, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with st.spinner("Running ML surrogate prediction..."):
        t0 = time.time()
        with torch.no_grad():
            pred_future_n = model(
                x_plan, x_ast,
                future_planets_xy=x_future_plan,
                future_len=future_len
            ).cpu().numpy()[0]
        t_ml = time.time() - t0

    # Unnormalize predictions back to AU and AU/yr
    pred_future = pred_future_n.copy()
    pred_future[:, 0:2] *= AST_SCALE_POS
    pred_future[:, 2:4] *= AST_SCALE_VEL

    # Metrics
    err_au = np.linalg.norm(pred_future[:, 0:2] - true_future[:, 0:2], axis=1)
    err = err_au * (AU_TO_KM if show_error_km else 1.0)
    unit = "km" if show_error_km else "AU"

    mean_err = float(err.mean())
    max_err = float(err.max())
    final_err = float(err[-1])

    r_mean_au = float(np.linalg.norm(true_future[:, 0:2], axis=1).mean())
    denom = r_mean_au * (AU_TO_KM if show_error_km else 1.0)
    rel_mean_pct = (mean_err / denom) * 100.0 if denom > 0 else float("nan")

    # Plots
    left, right = st.columns([1.2, 1.0])

    with left:
        fig = plt.figure(figsize=(9, 6))
        plt.plot(ast_true[:, 0], ast_true[:, 1], label="Asteroid (Physics truth)")
        plt.plot(true_future[:, 0], true_future[:, 1], label="Asteroid truth (forecast window)")
        plt.plot(pred_future[:, 0], pred_future[:, 1], "--", label="Asteroid (ML predicted)")

        if show_planets:
            for i, (name, _, _, _) in enumerate(PLANETS):
                xy = planet_pos_true[:, i, :]
                plt.plot(xy[:, 0], xy[:, 1], alpha=0.6, label=f"{name}")

        plt.scatter([0], [0], s=60, marker="*", label="Sun (origin)")
        plt.axis("equal")
        plt.xlabel("x (AU)")
        plt.ylabel("y (AU)")
        plt.title("Orbits: Physics vs ML Surrogate")
        plt.legend(loc="best", fontsize=8)
        st.pyplot(fig, clear_figure=True)

    with right:
        fig2 = plt.figure(figsize=(8, 4))
        plt.plot(err)
        plt.xlabel("Step into forecast window")
        plt.ylabel(f"Position error ({unit})")
        plt.title("Prediction Error Over Time")
        st.pyplot(fig2, clear_figure=True)

        st.markdown("### Results")
        st.write(f"**Mean position error:** {mean_err:,.2f} {unit}")
        st.write(f"**Max position error:** {max_err:,.2f} {unit}")
        st.write(f"**Final-step error:** {final_err:,.2f} {unit}")
        st.write(f"**Mean error as % of distance from Sun:** {rel_mean_pct:.2f}%")
        st.write(f"**Physics runtime:** {t_phys:.4f} s")
        st.write(f"**ML runtime:** {t_ml:.4f} s")
        if t_ml > 0:
            st.write(f"**Estimated speedup:** {t_phys / t_ml:.1f}×")
        st.caption("Speedup depends on CPU/GPU, time step, and trajectory length.")

st.divider()
st.markdown(
    """
### Science fair talking points (short)
- **Physics baseline:** I simulate gravity using an N-body numerical integrator (Leapfrog).
- **ML surrogate:** A **Graph Neural Network** summarizes gravitational “influences” from planets, and an **LSTM** predicts how the asteroid state changes over time.
- **Big upgrade:** The ML model can use **planet positions over time** during the prediction window (not frozen planets), improving realism and reducing error.
- **Why it matters:** ML can screen many asteroids quickly, then the most risky ones can be re-checked with high-precision physics.
"""
)

# -----------------------------
# Streamlit Cloud files (create in repo)
# -----------------------------
# requirements.txt (recommended):
# streamlit==1.32.2
# numpy==1.26.4
# matplotlib==3.8.4
# torch==2.2.2
#
# runtime.txt:
# python-3.11
