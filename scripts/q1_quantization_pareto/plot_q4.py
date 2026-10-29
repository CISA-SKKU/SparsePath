#!/usr/bin/env python
"""Q4 (M08) distributed placement A/B figure: rr vs lits (deep100M 4p, PQ off, T=64).
Numbers extracted from engine_report.txt of the two runs (verified)."""
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = [50, 100, 200, 400]
rr_qps = [23474, 14184, 7770, 4288]
lits_qps = [29674, 25063, 14815, 8251]
rr_ib = [75.7, 72.4, 72.3, 73.6]
lits_ib = [46.3, 45.3, 47.4, 45.0]
rr_remote, lits_remote = 74.7, 20.2
RR, LI = "#dc2626", "#16a34a"

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

# Panel A: QPS vs L
ax[0].plot(L, rr_qps, "s-", color=RR, label="round_robin (baseline)")
ax[0].plot(L, lits_qps, "^-", color=LI, label="lits (SparsePath)")
for i, l in enumerate(L):
    ax[0].annotate(f"{lits_qps[i]/rr_qps[i]:.2f}×", (l, lits_qps[i]),
                   textcoords="offset points", xytext=(0, 8), fontsize=8, color=LI)
ax[0].set_xlabel("L (recall matched: PQ off = exact)"); ax[0].set_ylabel("QPS (T=64)")
ax[0].set_title("(A) throughput — same recall, +1.8× QPS"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)

# Panel B: NIC IB_RX vs L
ax[1].plot(L, rr_ib, "s-", color=RR, label="round_robin")
ax[1].plot(L, lits_ib, "^-", color=LI, label="lits")
ax[1].axhline(80, ls=":", color="gray", alpha=0.7); ax[1].text(60, 81, "NIC ceiling ~80G/node", fontsize=7, color="gray")
ax[1].set_xlabel("L"); ax[1].set_ylabel("IB_RX (Gbps, T=64)")
ax[1].set_title("(B) NIC bandwidth — rr saturates, lits has headroom"); ax[1].set_ylim(0, 90)
ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)

# Panel C: remote ratio bars
bars = ax[2].bar(["round_robin\n(baseline)", "lits\n(SparsePath)"], [rr_remote, lits_remote],
                 color=[RR, LI], width=0.5, edgecolor="k")
for b, v in zip(bars, [rr_remote, lits_remote]):
    ax[2].text(b.get_x()+b.get_width()/2, v+1.5, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax[2].set_ylabel("remote vector access ratio (%)"); ax[2].set_ylim(0, 85)
ax[2].set_title("(C) remote-fetch COUNT — 3.7× fewer (lossless)"); ax[2].grid(axis="y", alpha=0.3)

fig.suptitle("Q4 (M08 / R5·W2): SparsePath placement reduces the COUNT of remote fetches "
             "(access-count axis) — lossless, orthogonal to quantization (packet-size axis)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
ts = time.strftime("%Y%m%d_%H%M%S")
out = f"/experiments/results/deep100M/q4_placement_ab/{ts}_q4_placement_qps_nic.png"
fig.savefig(out, dpi=130)
print(f"wrote {out}")
