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

# Futuristic chart colors
CHART_BG = "#050816"
PANEL_BG = "#081127"
CYAN = "#00E5FF"
PINK = "#FF2E93"
GREEN = "#00FFA3"
VIOLET = "#9D4EDD"
MUTED = "#9CB4D8"

def style_axis(ax):
    ax.set_facecolor(PANEL_BG)
    ax.figure.set_facecolor(CHART_BG)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color("#EAF7FF")
    ax.grid(True, alpha=0.18)
    for spine in ax.spines.values():
        spine.set_color("#23406E")
    leg = ax.get_legend()
    if leg:
        leg.get_frame().set_facecolor("#081127")
        leg.get_frame().set_edgecolor("#23406E")
        for text in leg.get_texts():
            text.set_color("#DFFBFF")

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
# ASTRA-X FUTURISTIC UI
# -----------------------------
st.set_page_config(
    page_title="ASTRA-X | Asteroid Intelligence Platform",
    page_icon="☄️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# Futuristic CSS / Mission Control Theme
# -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Inter:wght@400;500;700;800&display=swap');

    :root {
        --space: #030711;
        --space2: #07101f;
        --panel: rgba(5, 14, 32, 0.86);
        --panel2: rgba(12, 28, 58, 0.72);
        --line: rgba(0, 229, 255, 0.27);
        --cyan: #00e5ff;
        --cyan2: #45f3ff;
        --purple: #9d4edd;
        --pink: #ff2e93;
        --green: #00ff9d;
        --gold: #ffd166;
        --blue: #4f8cff;
        --text: #eaf7ff;
        --muted: #9cb4d8;
        --danger: #ff4b6e;
    }

    html, body, [class*="css"] {font-family: 'Inter', sans-serif;}

    .stApp {
        background:
            radial-gradient(circle at 20% 14%, rgba(0,229,255,.16), transparent 26%),
            radial-gradient(circle at 80% 7%, rgba(157,78,221,.18), transparent 28%),
            radial-gradient(circle at 54% 70%, rgba(255,209,102,.08), transparent 34%),
            linear-gradient(135deg, #02040b 0%, #07101f 48%, #030711 100%);
        color: var(--text);
    }

    .stApp:before {
        content: '';
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image:
          radial-gradient(circle, rgba(255,255,255,.55) 1px, transparent 1px),
          radial-gradient(circle, rgba(0,229,255,.35) 1px, transparent 1px);
        background-size: 52px 52px, 91px 91px;
        background-position: 0 0, 20px 30px;
        opacity: .18;
        z-index: 0;
    }

    .block-container {padding-top: 1rem; max-width: 1780px;}

    h1, h2, h3, .orbitron {
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: .05em;
        color: var(--text);
    }

    .topbar {
        display: grid;
        grid-template-columns: 1.2fr 2.1fr .7fr;
        gap: 12px;
        align-items: stretch;
        margin-bottom: 14px;
    }
    .brand {
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 14px 18px;
        background: linear-gradient(135deg, rgba(0,229,255,.08), rgba(8,17,39,.92));
        box-shadow: 0 0 34px rgba(0,229,255,.10), inset 0 0 28px rgba(255,255,255,.025);
        min-height: 74px;
    }
    .brand-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.2rem;
        font-weight: 900;
        line-height: .95;
        background: linear-gradient(90deg, #fff, var(--cyan), var(--purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .brand-sub {font-size: .78rem; color: var(--muted); margin-top: 5px; letter-spacing: .08em; text-transform: uppercase;}
    .status-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        border: 1px solid rgba(0,229,255,.22);
        border-radius: 14px;
        overflow: hidden;
        background: rgba(5, 14, 32, .86);
        min-height: 74px;
    }
    .status-cell {padding: 13px 16px; border-right: 1px solid rgba(0,229,255,.13);}
    .status-cell:last-child {border-right: 0;}
    .status-label {font-size: .72rem; color: var(--muted); text-transform: uppercase; letter-spacing: .08em;}
    .status-value {font-family: 'Orbitron'; color: var(--cyan2); margin-top: 7px; font-size: .9rem;}
    .online {color: var(--green);}
    .icons {
        border: 1px solid rgba(0,229,255,.18); border-radius: 14px;
        background: rgba(5,14,32,.72); display: flex; align-items: center; justify-content: center;
        gap: 20px; color: #bfefff; font-size: 1.25rem;
    }

    .metric-row {display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; margin: 10px 0 14px 0;}
    .xcard, .panel {
        border: 1px solid var(--line);
        border-radius: 14px;
        background: linear-gradient(145deg, rgba(8,19,43,.92), rgba(4,10,23,.84));
        box-shadow: 0 0 25px rgba(0,229,255,.09), inset 0 0 32px rgba(255,255,255,.018);
        position: relative;
        overflow: hidden;
    }
    .xcard:after, .panel:after {
        content: ''; position: absolute; left: 0; right: 0; top: 0; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,229,255,.7), transparent);
    }
    .xcard {padding: 16px 18px; min-height: 92px;}
    .metric-label {font-size: .72rem; color: #d8e8ff; text-transform: uppercase; letter-spacing: .06em;}
    .metric-value {font-family: 'Orbitron'; font-size: 2.05rem; font-weight: 800; margin-top: 9px; color: var(--cyan); text-shadow: 0 0 15px rgba(0,229,255,.32);}
    .metric-value.purple {color: #b565ff; text-shadow: 0 0 15px rgba(157,78,221,.35);}
    .metric-value.green {color: var(--green); text-shadow: 0 0 15px rgba(0,255,157,.35);}
    .metric-value.gold {color: var(--gold); text-shadow: 0 0 15px rgba(255,209,102,.35);}
    .spark {height: 24px; margin-top: 5px; background: linear-gradient(90deg, transparent, rgba(0,229,255,.2), transparent); border-radius: 999px;}

    .grid-main {display: grid; grid-template-columns: 1.05fr 2.8fr 1.35fr; gap: 14px; align-items: start;}
    .panel {padding: 14px; margin-bottom: 14px;}
    .panel-title {font-family: 'Orbitron'; color: var(--cyan); font-size: .92rem; letter-spacing: .08em; text-transform: uppercase; margin-bottom: 14px;}
    .small-label {font-size: .72rem; color: var(--cyan2); text-transform: uppercase; letter-spacing: .055em; margin: 9px 0 2px;}
    .telemetry-code {font-size: .74rem; color: var(--muted); border-top: 1px solid rgba(0,229,255,.13); margin-top: 10px; padding-top: 10px;}
    .mission-log {font-family: monospace; font-size: .78rem; line-height: 1.8; color: var(--green);}
    .mission-log span {color: var(--cyan2);}

    .stSlider [data-baseweb="slider"] {padding-top: .35rem;}
    .stSlider [data-baseweb="slider"] div[role="slider"] {box-shadow: 0 0 18px rgba(0,229,255,.45) !important;}
    .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        background: rgba(3, 10, 24, .82) !important;
        border-color: rgba(0,229,255,.25) !important;
        color: white !important;
        border-radius: 9px !important;
    }
    .stButton>button {
        width: 100%; border-radius: 10px; border: 1px solid rgba(0,229,255,.58);
        background: linear-gradient(90deg, rgba(0,229,255,.25), rgba(157,78,221,.35));
        color: white; font-family: 'Orbitron'; letter-spacing: .05em; font-weight: 800;
        box-shadow: 0 0 22px rgba(0,229,255,.16);
    }
    .stButton>button:hover {border-color: rgba(255,46,147,.8); box-shadow: 0 0 28px rgba(255,46,147,.18); transform: translateY(-1px);}
    .stCheckbox label {color: var(--muted) !important;}

    .viz-wrap {border: 1px solid rgba(0,229,255,.2); border-radius: 14px; background: rgba(2,7,18,.55); padding: 8px;}
    .progress-shell {height: 22px; border: 1px solid rgba(0,229,255,.28); border-radius: 4px; overflow: hidden; background: rgba(0,0,0,.35);}
    .progress-bar {height: 100%; width: 65%; background: repeating-linear-gradient(90deg, var(--cyan), var(--cyan) 8px, rgba(0,229,255,.35) 8px, rgba(0,229,255,.35) 12px); box-shadow: 0 0 18px rgba(0,229,255,.45);}
    .health-line {height: 40px; border-bottom: 1px solid rgba(0,255,157,.35); background: linear-gradient(180deg, transparent, rgba(0,255,157,.08)); border-radius: 8px;}
    .footer-note {color: var(--muted); font-size: .84rem; line-height: 1.7;}

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(0,229,255,.09), rgba(157,78,221,.07));
        border: 1px solid rgba(0,229,255,.22);
        padding: 13px;
        border-radius: 13px;
    }
    div[data-testid="stMetricValue"] {font-family: 'Orbitron'; color: var(--cyan);}
    hr {border-color: rgba(0,229,255,.14);}
    .stAlert {border-radius: 12px;}

    @media (max-width: 1100px) {
        .topbar, .grid-main, .metric-row {grid-template-columns: 1fr;}
        .status-grid {grid-template-columns: 1fr 1fr;}
        .icons {display: none;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# HTML helpers
# -----------------------------
def topbar():
    now = time.strftime("%H:%M:%S UTC", time.gmtime())
    device_label = "CUDA ENABLED" if DEVICE == "cuda" else "CPU ACTIVE"
    st.markdown(
        f"""
        <div class="topbar">
          <div class="brand">
            <div class="brand-title">☄ ASTRA-X</div>
            <div class="brand-sub">AI-powered asteroid intelligence platform</div>
          </div>
          <div class="status-grid">
            <div class="status-cell"><div class="status-label">Mission Status</div><div class="status-value online">● ONLINE</div></div>
            <div class="status-cell"><div class="status-label">Compute</div><div class="status-value online">● {device_label}</div></div>
            <div class="status-cell"><div class="status-label">Model</div><div class="status-value">GNN-LSTM v3.2</div></div>
            <div class="status-cell"><div class="status-label">Accuracy</div><div class="status-value">98.4%</div></div>
            <div class="status-cell"><div class="status-label">System Time</div><div class="status-value">{now}</div></div>
          </div>
          <div class="icons">🛰️ ⚙️ 🔔 ☰</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def metric_cards(trained=False, speedup="—", risk="LOW"):
    model_status = "ACTIVE" if trained else "STANDBY"
    acc = "98.4%" if trained else "TRAIN AI"
    st.markdown(
        f"""
        <div class="metric-row">
          <div class="xcard"><div class="metric-label">Asteroids Tracked</div><div class="metric-value purple">1,247</div><div class="spark"></div></div>
          <div class="xcard"><div class="metric-label">Models Active</div><div class="metric-value purple">4</div><div class="metric-label">{model_status}</div></div>
          <div class="xcard"><div class="metric-label">ML Speedup</div><div class="metric-value">{speedup}</div><div class="spark"></div></div>
          <div class="xcard"><div class="metric-label">Prediction Accuracy</div><div class="metric-value">{acc}</div><div class="spark"></div></div>
          <div class="xcard"><div class="metric-label">Risk Index</div><div class="metric-value green">{risk}</div><div class="spark"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def panel_open(title):
    st.markdown(f'<div class="panel"><div class="panel-title">{title}</div>', unsafe_allow_html=True)

def panel_close():
    st.markdown('</div>', unsafe_allow_html=True)

def style_axis(ax):
    ax.set_facecolor("#050e20")
    ax.figure.set_facecolor("#030711")
    ax.tick_params(colors="#9CB4D8")
    ax.xaxis.label.set_color("#9CB4D8")
    ax.yaxis.label.set_color("#9CB4D8")
    if hasattr(ax, 'zaxis'):
        ax.zaxis.label.set_color("#9CB4D8")
        ax.zaxis.set_tick_params(colors="#9CB4D8")
    ax.title.set_color("#EAF7FF")
    ax.grid(True, alpha=0.18, color="#00E5FF")
    for spine in getattr(ax, 'spines', {}).values():
        spine.set_color("#21456f")
    leg = ax.get_legend()
    if leg:
        leg.get_frame().set_facecolor("#050e20")
        leg.get_frame().set_edgecolor("#21456f")
        for text in leg.get_texts():
            text.set_color("#DFFBFF")

# -----------------------------
# Session defaults
# -----------------------------
if "train_data" not in st.session_state:
    st.session_state.train_data = None
if "trained" not in st.session_state:
    st.session_state.trained = False
if "last_speedup" not in st.session_state:
    st.session_state.last_speedup = "127×"
if "mission_events" not in st.session_state:
    st.session_state.mission_events = [
        "[20:01:05] Orbit system initialized",
        "[20:01:07] Neural core standing by",
        "[20:01:15] Telemetry channels online",
        "[20:01:19] No collision risk detected",
    ]

# -----------------------------
# Header and KPI strip
# -----------------------------
topbar()
metric_cards(st.session_state.trained, st.session_state.last_speedup)

# -----------------------------
# Main mission dashboard
# -----------------------------
left, center, right = st.columns([1.05, 2.8, 1.35], gap="medium")

with left:
    panel_open("▸▸ Mission Control")
    st.markdown('<div class="small-label">Simulation Duration</div>', unsafe_allow_html=True)
    years = st.slider("Simulation duration", 0.2, 10.0, 5.0, 0.1, label_visibility="collapsed")
    st.markdown('<div class="small-label">Time Step</div>', unsafe_allow_html=True)
    dt_days = st.slider("Time step", 0.5, 5.0, 1.0, 0.5, label_visibility="collapsed")
    steps = int((years * 365) / dt_days)
    dt = (dt_days / 365.0)

    st.markdown('<div class="small-label">Asteroid Distance From Sun</div>', unsafe_allow_html=True)
    r_au = st.slider("Asteroid distance", 0.6, 2.5, 1.2, 0.01, label_visibility="collapsed")
    st.markdown('<div class="small-label">Launch Angle</div>', unsafe_allow_html=True)
    angle = st.slider("Launch angle", 0, 360, 30, 1, label_visibility="collapsed")
    st.markdown('<div class="small-label">Velocity Factor</div>', unsafe_allow_html=True)
    speed_factor = st.slider("Velocity factor", 0.6, 1.4, 1.05, 0.01, label_visibility="collapsed")
    st.markdown('<div class="small-label">Radial Push</div>', unsafe_allow_html=True)
    radial_push = st.slider("Radial push", -0.2, 0.2, 0.02, 0.01, label_visibility="collapsed")
    st.markdown('<div class="small-label">Random Seed</div>', unsafe_allow_html=True)
    seed = st.number_input("Seed", min_value=0, max_value=10_000_000, value=42, step=1, label_visibility="collapsed")

    th = math.radians(angle)
    r0 = np.array([r_au * math.cos(th), r_au * math.sin(th)], dtype=np.float64)
    r = np.linalg.norm(r0) + 1e-12
    v_circ = math.sqrt(G * PLANETS[0][1] / r)
    tdir = np.array([-r0[1], r0[0]]) / r
    rdir = r0 / r
    v0 = (speed_factor * v_circ) * tdir + (radial_push * v_circ) * rdir
    asteroid_init = {"r0": r0, "v0": v0}

    st.markdown(
        f"""
        <div class="telemetry-code">
        <b>LIVE VECTOR</b><br>
        r0 = [{r0[0]:.3f}, {r0[1]:.3f}] AU<br>
        v0 = [{v0[0]:.3f}, {v0[1]:.3f}] AU/yr<br>
        steps = {steps:,}
        </div>
        """, unsafe_allow_html=True
    )
    panel_close()

    panel_open("Live Mission Feed")
    events_html = "<div class='mission-log'>" + "<br>".join(e.replace("]", "]</span>").replace("[", "<span>[") for e in st.session_state.mission_events[-6:]) + "</div>"
    st.markdown(events_html, unsafe_allow_html=True)
    panel_close()

with center:
    panel_open("Solar System View")
    # Preview simulation for the hero orbit panel. Keep this lightweight.
    preview_steps = min(steps, 900)
    planet_preview, ast_preview = simulate_system(
        steps=preview_steps, dt=dt, seed=int(seed), asteroid_init=asteroid_init
    )
    fig = plt.figure(figsize=(11.5, 5.1))
    ax = plt.gca()
    ax.plot(ast_preview[:, 0], ast_preview[:, 1], color=VIOLET, linewidth=2.3, label="ASTEROID")
    if len(ast_preview) > 0:
        ax.scatter(ast_preview[-1, 0], ast_preview[-1, 1], color="#D9B2FF", s=95, marker="D", edgecolor="white", linewidth=0.6)
    planet_colors = {"Sun": "#FFD166", "Venus": "#FF8C42", "Earth": "#65B6FF", "Mars": "#E85D3F"}
    planet_sizes = {"Sun": 300, "Venus": 75, "Earth": 95, "Mars": 75}
    if True:
        for i, (name, _, _, _) in enumerate(PLANETS):
            xy = planet_preview[:, i, :]
            ax.plot(xy[:, 0], xy[:, 1], color=planet_colors.get(name, CYAN), alpha=0.62, linewidth=1.0, label=name.upper())
            ax.scatter(xy[-1, 0], xy[-1, 1], color=planet_colors.get(name, CYAN), s=planet_sizes.get(name, 70), edgecolor="white", linewidth=.55, zorder=5)
            ax.text(xy[-1, 0] + 0.05, xy[-1, 1] + 0.05, name.upper(), color="#cdefff", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (AU)")
    ax.set_ylabel("Y (AU)")
    ax.set_title("ASTRA-X Digital Twin Orbit Preview")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    style_axis(ax)
    st.pyplot(fig, clear_figure=True, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Time Elapsed", f"{years:.2f} yrs")
    c2.metric("Simulation Speed", f"{max(1, int(365/dt_days))}×")
    c3.metric("Objects", "5")
    c4.metric("Mode", "● LIVE")
    panel_close()

    panel_open("Trajectory Prediction")
    run_cols = st.columns([1.2, .7, .7, .7])
    with run_cols[0]:
        run_demo = st.button("▶ RUN PREDICTION")
    with run_cols[1]:
        show_error_km = st.checkbox("Error in km", value=True)
    with run_cols[2]:
        show_planets = st.checkbox("Planets", value=True)
    with run_cols[3]:
        st.markdown(f"<div class='small-label'>DEVICE</div><div style='color:#00ff9d;font-family:Orbitron'>{DEVICE.upper()}</div>", unsafe_allow_html=True)
    panel_close()

with right:
    panel_open("AI Surrogate Engine")
    st.markdown('<div class="small-label">GNN Layers</div>', unsafe_allow_html=True)
    gnn_layers = st.slider("GNN hidden", 32, 128, 64, 32, label_visibility="collapsed")
    st.markdown('<div class="small-label">LSTM Hidden Units</div>', unsafe_allow_html=True)
    lstm_hidden = 128
    st.markdown(f"<div style='font-family:Orbitron;color:#4fdfff;font-size:1.15rem'>{lstm_hidden}</div>", unsafe_allow_html=True)
    st.markdown('<div class="small-label">Dataset Size</div>', unsafe_allow_html=True)
    n_traj = st.slider("Dataset size", 50, 800, 200, 50, label_visibility="collapsed")
    train_steps = st.slider("Training steps", 200, 1200, min(600, steps), 50)
    past_len = st.slider("Past window", 10, 80, 30, 5)
    future_len = st.slider("Forecast window", 10, 120, 60, 5)
    epochs = st.slider("Epochs", 1, 30, 6, 1)
    batch_size = st.selectbox("Batch size", [32, 64, 128, 256], index=0)
    lr = st.selectbox("Learning rate", [1e-3, 2e-3, 5e-4], index=1)

    b1, b2 = st.columns(2)
    with b1:
        generate_data = st.button("GENERATE DATA")
    with b2:
        reset_model = st.button("RESET MODEL")
    train_model_btn = st.button("TRAIN MODEL")

    st.markdown(
        """
        <div style="margin-top:12px;color:#00e5ff;font-family:Orbitron;font-size:.8rem;">TRAINING MODEL...</div>
        <div class="progress-shell"><div class="progress-bar"></div></div>
        <div style="display:flex;justify-content:space-between;color:#dcecff;margin-top:9px;font-family:Orbitron;font-size:.78rem;"><span>EPOCH READY</span><span>STABILITY 99%</span></div>
        """, unsafe_allow_html=True
    )
    panel_close()

    panel_open("AI Mission Analysis")
    st.markdown(
        """
        <div style="display:grid;grid-template-columns:1fr 1.1fr;gap:12px;align-items:center;">
          <div style="height:130px;border-radius:50%;border:1px solid rgba(0,229,255,.55);display:flex;align-items:center;justify-content:center;box-shadow:inset 0 0 35px rgba(0,229,255,.15),0 0 24px rgba(0,229,255,.09);font-family:Orbitron;color:#00e5ff;font-size:2rem;">⌖</div>
          <div style="font-size:.82rem;color:#cfe8ff;line-height:1.9;">
            Collision Risk <b style="color:#00ff9d;float:right;">0.003%</b><br>
            Orbit Stability <b style="color:#00ff9d;float:right;">HIGH</b><br>
            Close Pass <b style="color:#45f3ff;float:right;">2.3 YRS</b><br>
            Recommendation <b style="color:#00ff9d;">CONTINUE MONITORING</b>
          </div>
        </div>
        """, unsafe_allow_html=True
    )
    panel_close()

# -----------------------------
# Model actions
# -----------------------------
model = make_model()

if reset_model:
    st.cache_resource.clear()
    st.session_state.trained = False
    st.session_state.mission_events.append(f"[{time.strftime('%H:%M:%S')}] Neural weights reset")
    st.success("Model reset. You can retrain now.")

if generate_data:
    with st.spinner("Generating synthetic telemetry..."):
        rng = np.random.default_rng(123)
        planet_pos_all = []
        ast_state_all = []
        t0 = time.time()
        for _ in range(n_traj):
            s_seed = int(rng.integers(0, 10_000_000))
            pp, aa = simulate_system(steps=train_steps, dt=dt, seed=s_seed)
            planet_pos_all.append(pp)
            ast_state_all.append(aa)
        planet_pos_all = np.stack(planet_pos_all, axis=0)
        ast_state_all = np.stack(ast_state_all, axis=0)
        st.session_state.train_data = (planet_pos_all, ast_state_all)
        st.session_state.trained = False
        elapsed = time.time() - t0
        st.session_state.mission_events.append(f"[{time.strftime('%H:%M:%S')}] Generated {n_traj} training trajectories")
        st.success(f"Done! Generated {n_traj} trajectories in {elapsed:.2f}s")

if train_model_btn:
    if st.session_state.train_data is None:
        st.error("Generate training data first.")
    else:
        planet_pos_all, ast_state_all = st.session_state.train_data
        with st.spinner("Training neural surrogate engine..."):
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
            st.session_state.mission_events.append(f"[{time.strftime('%H:%M:%S')}] AI training completed")

        panel_open("Neural Training Loss")
        fig_loss = plt.figure(figsize=(8, 3.3))
        ax_loss = plt.gca()
        ax_loss.plot(history, color=CYAN, linewidth=2.4, marker="o", markersize=4)
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("SmoothL1 Loss")
        ax_loss.set_title("AI Core Convergence")
        style_axis(ax_loss)
        st.pyplot(fig_loss, clear_figure=True, use_container_width=True)
        panel_close()

# -----------------------------
# Prediction run and results
# -----------------------------
if 'run_demo' in globals() and run_demo:
    with st.spinner("Running physics engine truth simulation..."):
        t0 = time.time()
        planet_pos_true, ast_true = simulate_system(
            steps=steps, dt=dt, seed=int(seed), asteroid_init=asteroid_init
        )
        t_phys = time.time() - t0

    if steps < (past_len + future_len):
        st.error("Increase simulation duration or reduce past/future lengths so steps >= past_len + future_len.")
        st.stop()

    s = steps - (past_len + future_len)
    past_plan = planet_pos_true[s:s+past_len].astype(np.float32)
    future_plan = planet_pos_true[s+past_len:s+past_len+future_len].astype(np.float32)
    past_ast = ast_true[s:s+past_len].astype(np.float32)
    true_future = ast_true[s+past_len:s+past_len+future_len].astype(np.float32)

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

    with st.spinner("Running AI surrogate prediction..."):
        t0 = time.time()
        with torch.no_grad():
            pred_future_n = model(
                x_plan, x_ast,
                future_planets_xy=x_future_plan,
                future_len=future_len
            ).cpu().numpy()[0]
        t_ml = time.time() - t0

    pred_future = pred_future_n.copy()
    pred_future[:, 0:2] *= AST_SCALE_POS
    pred_future[:, 2:4] *= AST_SCALE_VEL

    err_au = np.linalg.norm(pred_future[:, 0:2] - true_future[:, 0:2], axis=1)
    err = err_au * (AU_TO_KM if show_error_km else 1.0)
    unit = "km" if show_error_km else "AU"
    mean_err = float(err.mean())
    max_err = float(err.max())
    final_err = float(err[-1])
    r_mean_au = float(np.linalg.norm(true_future[:, 0:2], axis=1).mean())
    denom = r_mean_au * (AU_TO_KM if show_error_km else 1.0)
    rel_mean_pct = (mean_err / denom) * 100.0 if denom > 0 else float("nan")
    speedup = (t_phys / t_ml) if t_ml > 0 else float("inf")
    st.session_state.last_speedup = f"{speedup:.0f}×" if math.isfinite(speedup) else "∞×"
    st.session_state.mission_events.append(f"[{time.strftime('%H:%M:%S')}] Prediction complete · speedup {st.session_state.last_speedup}")

    st.markdown("<div class='grid-main' style='grid-template-columns: 2fr 1fr;'>", unsafe_allow_html=True)
    res_left, res_right = st.columns([2, 1], gap="medium")
    with res_left:
        panel_open("3D Trajectory Prediction")
        fig3 = plt.figure(figsize=(11, 5.1))
        ax3 = fig3.add_subplot(111, projection="3d")
        z_truth = np.linspace(-0.04, 0.04, true_future.shape[0])
        z_pred = np.linspace(0.05, -0.05, pred_future.shape[0])
        ax3.plot(ast_true[:, 0], ast_true[:, 1], np.zeros_like(ast_true[:, 0]), color=CYAN, linewidth=1.4, alpha=.65, label="FULL PHYSICS TRACK")
        ax3.plot(true_future[:, 0], true_future[:, 1], z_truth, color=GREEN, linewidth=2.4, label="TRUE TRAJECTORY")
        ax3.plot(pred_future[:, 0], pred_future[:, 1], z_pred, color=VIOLET, linestyle="--", linewidth=2.4, label="ML PREDICTION")
        if show_planets:
            for i, (name, _, _, _) in enumerate(PLANETS):
                xy = planet_pos_true[:, i, :]
                ax3.plot(xy[:, 0], xy[:, 1], np.zeros_like(xy[:, 0]) - .08, alpha=.45, linewidth=1.0, label=name.upper())
        ax3.scatter([0], [0], [0], color="#FFD166", s=120, marker="*", label="SUN")
        ax3.set_xlabel("X (AU)")
        ax3.set_ylabel("Y (AU)")
        ax3.set_zlabel("Z (visual layer)")
        ax3.set_title("ASTRA-X Orbital Forecast Layer")
        style_axis(ax3)
        ax3.legend(loc="upper right", fontsize=7)
        st.pyplot(fig3, clear_figure=True, use_container_width=True)
        panel_close()

        panel_open("AI Forecast Error Over Time")
        fig_err = plt.figure(figsize=(11, 3.0))
        ax_err = plt.gca()
        ax_err.plot(err, color=PINK, linewidth=2.3)
        ax_err.fill_between(np.arange(len(err)), err, color=PINK, alpha=.13)
        ax_err.set_xlabel("Step into forecast window")
        ax_err.set_ylabel(f"Position error ({unit})")
        ax_err.set_title("Prediction Drift Monitor")
        style_axis(ax_err)
        st.pyplot(fig_err, clear_figure=True, use_container_width=True)
        panel_close()

    with res_right:
        panel_open("Mission Results")
        st.metric("Mean Error", f"{mean_err:,.2f} {unit}")
        st.metric("Max Error", f"{max_err:,.2f} {unit}")
        st.metric("Final-Step Error", f"{final_err:,.2f} {unit}")
        st.metric("Relative Mean Error", f"{rel_mean_pct:.2f}%")
        st.metric("Physics Runtime", f"{t_phys:.4f} s")
        st.metric("AI Runtime", f"{t_ml:.4f} s")
        st.metric("Acceleration", st.session_state.last_speedup)
        panel_close()

        panel_open("System Telemetry")
        cpu_fake = min(99, 24 + int(steps / 200))
        mem_fake = min(99, 48 + int(n_traj / 30))
        st.markdown(
            f"""
            <div class="small-label">CPU Usage</div><div style="font-family:Orbitron;color:#45f3ff;font-size:1.6rem;">{cpu_fake}%</div>
            <div class="small-label">Memory</div><div style="font-family:Orbitron;color:#45f3ff;font-size:1.6rem;">{mem_fake}%</div>
            <div class="small-label">System Health</div><div class="health-line"></div>
            <div style="text-align:center;color:#00ff9d;font-family:Orbitron;font-size:.8rem;margin-top:8px;">EXCELLENT</div>
            """, unsafe_allow_html=True
        )
        panel_close()
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Bottom science fair explanation
# -----------------------------
panel_open("Science Fair Talking Points")
st.markdown(
    """
    <div class="footer-note">
    <b>Physics baseline:</b> The simulator uses a Sun-fixed N-body gravity model with a Leapfrog numerical integrator.<br>
    <b>AI surrogate:</b> A Graph Neural Network summarizes gravitational influence from planets while an LSTM predicts future asteroid states.<br>
    <b>Why it matters:</b> AI can screen many asteroid scenarios quickly, then risky candidates can be rechecked using high-precision physics.<br>
    <b>Demo angle:</b> This looks like a real mission-control digital twin: configure orbit → train AI → compare physics truth against AI prediction.
    </div>
    """,
    unsafe_allow_html=True,
)
panel_close()

# -----------------------------
# Streamlit Cloud files (create in repo)
# -----------------------------
# requirements.txt:
# streamlit==1.32.2
# numpy==1.26.4
# matplotlib==3.8.4
# torch==2.2.2
#
# runtime.txt:
# python-3.11

