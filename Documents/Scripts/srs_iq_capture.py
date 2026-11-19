#!/usr/bin/env python3
"""
Receive SRS channel estimates from the patched srsRAN gNB (srs_iq_udp.h) and save them.
"""
import argparse, socket, struct, sys, time
import numpy as np

HDR = struct.Struct("<4sHHHBBBBBBHHffffd")  # 44 bytes, matches srs_iq::header
FIELDS = ("rnti", "sfn", "slot", "numerology", "rx_port", "start_symbol",
          "nof_symbols", "comb", "first_sc", "n_sc",
          "ta_us", "epre_db", "rsrp_db", "noise_var", "t")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=50000)
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--out", default="srs_run")
    a = p.parse_args()

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
    s.bind(("127.0.0.1", a.port))
    s.settimeout(0.5)

    meta = {k: [] for k in FIELDS}
    H, bad = [], 0
    t_end = time.time() + a.seconds
    print(f"listening on 127.0.0.1:{a.port} for {a.seconds:.0f} s (Ctrl+C to stop early)")
    try:
        while time.time() < t_end:
            try:
                pkt = s.recv(65535)
            except socket.timeout:
                continue
            if len(pkt) < HDR.size or pkt[:4] != b"SRSQ":
                bad += 1
                continue
            v = HDR.unpack_from(pkt)
            n_sc = v[11]
            iq = np.frombuffer(pkt, np.complex64, n_sc, HDR.size)
            for k, x in zip(FIELDS, v[2:]):
                meta[k].append(x)
            H.append(iq.copy())
            if len(H) % 200 == 0:
                print(f"\r{len(H)} records  rnti={v[2]:#x}  ta={v[12]:.2f} us  "
                      f"rsrp={v[14]:.1f} dB", end="", flush=True)
    except KeyboardInterrupt:
        print("\nstopped early")

    if not H:
        sys.exit("\nno packets. Is the gNB running with SRS_IQ_PORT set, and SRS enabled for the UE?")

    n_sc = max(len(h) for h in H)
    Hm = np.zeros((len(H), n_sc), np.complex64)
    for i, h in enumerate(H):
        Hm[i, :len(h)] = h
    out = {k: np.asarray(v) for k, v in meta.items()}
    out["t"] = out["t"] - out["t"][0]
    np.savez_compressed(a.out + ".npz", H=Hm, **out)

    dur = out["t"][-1]
    print(f"\nsaved {len(H)} records x {n_sc} subcarriers, {dur:.1f} s -> {a.out}.npz")
    print(f"rate {len(H)/max(dur,1e-9):.0f}/s   RNTIs {len(set(out['rnti']))}   "
          f"bad packets {bad}   median RSRP {np.median(out['rsrp_db']):.1f} dB")


if __name__ == "__main__":
    main()
