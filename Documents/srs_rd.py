"""Simulation of uplink-SRS based bistatic range-Doppler sensing.

Chain: SRS (Zadoff-Chu on a comb) -> channel with direct path + reflections
-> per-occasion timing/phase offsets -> least-squares CSI -> direct-path
calibration -> IFFT over subcarriers (range) / FFT over SRS occasions
(Doppler) -> static-clutter removal -> 2-D CA-CFAR -> range-Doppler map.

"""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

C = 3e8

# ---- parameters (edit) ---------------------------------------------------
FC = 3.5e9          # carrier frequency (Hz)
SCS = 30e3          # NR subcarrier spacing (Hz)
BW = 80e6           # SRS-occupied bandwidth (Hz)
COMB = 2            # SRS comb factor (every COMB-th subcarrier)
T_SRS = 5e-3        # SRS period (s)
N_OCC = 128         # number of SRS occasions processed
SNR_DB = 10         # per-subcarrier SNR of the direct path
SEED = 1

# Paths: (excess bistatic range over direct path [m], bistatic Doppler [Hz], rel. amplitude)
PATHS = [
    (0.0, 0.0, 1.0),     # direct path (UE -> gNB)
    (30.0, 25.0, 0.10),  # moving target
    (60.0, 0.0, 0.15),   # static clutter (wall)
]
# --------------------------------------------------------------------------

rng = np.random.default_rng(SEED)
K = int(BW / (SCS * COMB))            # number of SRS subcarriers
df = SCS * COMB                       # effective subcarrier spacing
k = np.arange(K)
n = np.arange(N_OCC)

# Zadoff-Chu sequence (root 25) as the known SRS
Nzc = K if K % 2 else K + 1
x = np.exp(-1j * np.pi * 25 * np.arange(Nzc) * (np.arange(Nzc) + 1) / Nzc)[:K]

# True channel H[k, n]
H = np.zeros((K, N_OCC), complex)
for dR, fD, a in PATHS:
    tau = dR / C
    H += a * np.exp(-2j * np.pi * k[:, None] * df * tau) * np.exp(2j * np.pi * fD * n[None, :] * T_SRS)

# Per-occasion hardware offsets: timing (linear phase over k) and common phase
timing = rng.uniform(-30e-9, 30e-9, N_OCC)          # +-30 ns jitter
cphase = rng.uniform(0, 2 * np.pi, N_OCC)           # random phase each occasion
off = np.exp(-2j * np.pi * k[:, None] * df * timing[None, :]) * np.exp(1j * cphase[None, :])

# Received SRS Y = H * X * offsets + noise
sigma = 10 ** (-SNR_DB / 20)
noise = sigma * (rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape)) / np.sqrt(2)
Y = H * off * x[:, None] + noise

# 1) least-squares CSI
Hh = Y / x[:, None]

# 2) direct-path calibration: the direct path is the strongest component. It is
#    isolated with a gate around its range peak, then its delay (phase slope over
#    subcarriers) and common phase are estimated per occasion and removed. This
#    cancels timing jitter and oscillator phase without being biased by reflections.
M = 4 * K                                           # zero-padded range FFT
GATE = 12                                           # gate half-width (range bins)
prof0 = np.fft.ifft(Hh, n=M, axis=0)
Hc = np.empty_like(Hh)
for i in range(N_OCC):
    m0 = np.argmax(np.abs(prof0[:, i]))
    idx = np.arange(m0 - GATE, m0 + GATE + 1) % M
    gated = np.zeros(M, complex)
    gated[idx] = prof0[idx, i]
    hd = np.fft.fft(gated)[:K] * M / K              # direct path only, back in frequency
    slope = np.angle(np.sum(hd[1:] * np.conj(hd[:-1])))
    h = Hh[:, i] * np.exp(-1j * slope * k)
    h = h * np.exp(-1j * np.angle(np.sum(hd * np.exp(-1j * slope * k))))
    Hc[:, i] = h

# 3) range (IFFT over k) and Doppler (FFT over occasions)
rprof = np.fft.ifft(Hc * np.hanning(K)[:, None], n=M, axis=0)
rng_axis = np.arange(M) / (M * df) * C              # bistatic excess range (m)
sel = rng_axis <= 150
rprof = rprof[sel]
rng_axis = rng_axis[sel]

# 4) static-clutter removal (subtract mean over slow time)
rprof_d = rprof - rprof.mean(axis=1, keepdims=True)

RD = np.fft.fftshift(np.fft.fft(rprof_d * np.hanning(N_OCC)[None, :], axis=1), axes=1)
dop_axis = np.fft.fftshift(np.fft.fftfreq(N_OCC, T_SRS))
P = np.abs(RD) ** 2

# 5) 2-D cell-averaging CFAR
def ca_cfar(p, guard=(2, 2), train=(8, 6), pfa=1e-4):
    gr, gd = guard
    tr, td = train
    full = np.ones((2 * (gr + tr) + 1, 2 * (gd + td) + 1))
    inner = np.ones((2 * gr + 1, 2 * gd + 1))
    n_train = full.sum() - inner.sum()
    noise_est = (convolve2d(p, full, "same", boundary="symm") - convolve2d(p, inner, "same", boundary="symm")) / n_train
    alpha = n_train * (pfa ** (-1 / n_train) - 1)
    return p > alpha * noise_est

det = ca_cfar(P)
ri, di = np.nonzero(det)
# keep local maxima only
keep = [(r, d) for r, d in zip(ri, di) if P[r, d] == P[max(r - 3, 0):r + 4, max(d - 3, 0):d + 4].max()]

print(f"K = {K} SRS subcarriers, effective spacing {df/1e3:.0f} kHz")
print(f"Range resolution c/B = {C / BW:.2f} m, range bin spacing {rng_axis[1] - rng_axis[0]:.2f} m")
print(f"Doppler resolution 1/(N*T) = {1 / (N_OCC * T_SRS):.2f} Hz, unambiguous +-{1 / (2 * T_SRS):.0f} Hz")
print("Detections (excess bistatic range m, Doppler Hz):")
for r, d in sorted(keep, key=lambda t: -P[t]):
    print(f"  {rng_axis[r]:7.1f} m   {dop_axis[d]:7.1f} Hz   {10 * np.log10(P[r, d] / np.median(P)):5.1f} dB")

# 6) plot
fig, ax = plt.subplots(figsize=(4.2, 3.2))
im = ax.imshow(10 * np.log10(P / P.max() + 1e-12), aspect="auto", origin="lower", cmap="gray_r", vmin=-50, vmax=0,
               extent=[dop_axis[0], dop_axis[-1], rng_axis[0], rng_axis[-1]])
for r, d in keep:
    ax.plot(dop_axis[d], rng_axis[r], "o", mfc="none", mec="r", ms=8)
ax.set_xlabel("Doppler (Hz)")
ax.set_ylabel("Excess bistatic range (m)")
ax.set_title("range-Doppler map")
fig.colorbar(im, label="dB")
fig.tight_layout()
fig.savefig("rd_map_sim.png", dpi=200)
print("saved rd_map_sim.png")
