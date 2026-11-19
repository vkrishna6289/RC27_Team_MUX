#!/usr/bin/env python3
"""
Plot SRS channel estimates saved by srs_iq_capture.py
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("--rnti", type=lambda x: int(x, 0), default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--max-points", type=int, default=200000)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    d = np.load(a.npz)
    H, rnti, t = d["H"], d["rnti"], d["t"]
    sel = np.ones(len(H), bool)
    if a.port is not None:
        sel &= d["rx_port"] == a.port
    if a.rnti is None:
        vals, cnt = np.unique(rnti[sel], return_counts=True)
        a.rnti = int(vals[cnt.argmax()])
    sel &= rnti == a.rnti
    H, t = H[sel], t[sel]
    H = H[:, np.abs(H).sum(0) > 0]              # drop zero padding
    if len(H) == 0:
        raise SystemExit("no records for that RNTI/port")
    print(f"RNTI {a.rnti:#x}: {len(H)} records, {H.shape[1]} subcarriers, {t[-1]-t[0]:.1f} s")

    scale = np.median(np.abs(H))
    Hn = H / scale
    cp = np.exp(-1j * np.angle(Hn.mean(axis=1, keepdims=True)))
    Hc = Hn * cp

    def thin(x, n):
        x = x.ravel()
        return x if len(x) <= n else x[np.random.default_rng(0).choice(len(x), n, replace=False)]

    fig, ax = plt.subplots(2, 2, figsize=(11, 10))
    lim = 1.1 * np.percentile(np.abs(Hn), 99.5)

    for a_, z, title in ((ax[0, 0], Hn, "Raw IQ"), (ax[0, 1], Hc, "IQ, common phase removed")):
        z = thin(z, a.max_points)
        a_.scatter(z.real, z.imag, s=1, alpha=0.3)
        a_.axhline(0, c="k", lw=0.6); a_.axvline(0, c="k", lw=0.6)
        a_.set_xlim(-lim, lim); a_.set_ylim(-lim, lim); a_.set_aspect("equal")
        a_.set_xlabel("I"); a_.set_ylabel("Q"); a_.set_title(title)

    im = ax[1, 0].imshow(20 * np.log10(np.abs(Hn).T + 1e-9), aspect="auto", origin="lower",
                         extent=[t[0], t[-1], 0, H.shape[1]], cmap="viridis")
    ax[1, 0].set_xlabel("time (s)"); ax[1, 0].set_ylabel("SRS subcarrier index")
    ax[1, 0].set_title("|H| (dB, normalised)")
    fig.colorbar(im, ax=ax[1, 0])

    k = max(1, len(Hc) // 3000)
    z = Hc[::k]
    tt = np.repeat(t[::k], z.shape[1])
    sc = ax[1, 1].scatter(z.real.ravel(), z.imag.ravel(), c=tt, s=1, cmap="plasma", alpha=0.4)
    ax[1, 1].axhline(0, c="k", lw=0.6); ax[1, 1].axvline(0, c="k", lw=0.6)
    ax[1, 1].set_xlim(-lim, lim); ax[1, 1].set_ylim(-lim, lim); ax[1, 1].set_aspect("equal")
    ax[1, 1].set_title("Common phase removed, coloured by time")
    fig.colorbar(sc, ax=ax[1, 1], label="time (s)")

    fig.suptitle(f"{a.npz}  RNTI {a.rnti:#x}  ({len(H)} SRS records)")
    fig.tight_layout()
    out = a.out or a.npz.replace(".npz", "_iq.png")
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
