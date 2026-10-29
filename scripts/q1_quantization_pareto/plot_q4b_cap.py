#!/usr/bin/env python
"""Q4b / M10 CAP ablation figure (deep100M 4p, T=64, L=100).
Shows PQ filter = fetch-COUNT reduction (remote%) ; batched fetch = LATENCY hiding
(us/call) — distinct mechanisms, largely additive. Numbers from engine_report.txt."""
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

cfg = ["rr\nbaseline", "+lits\n(placement)", "+PQ filter\n(count↓)", "+batched\n(latency↓)", "full\n(lits+PQ+batch)"]
qps = [14184, 25063, 28571, 31250, 34014]
remote = [74.7, 20.2, 3.4, 20.3, 3.6]
uscall = [20.23, 6.54, 2.80, 3.61, 1.54]
COL = ["#dc2626", "#2563eb", "#16a34a", "#9333ea", "#111827"]
x = np.arange(len(cfg))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.7))

# A: QPS
b = ax[0].bar(x, qps, color=COL, edgecolor="k", width=0.62)
for i, v in enumerate(qps):
    ax[0].text(i, v+400, f"{v/1000:.1f}K", ha="center", fontsize=9, fontweight="bold")
ax[0].text(4, 34014/2, f"{qps[4]/qps[0]:.1f}× vs rr", ha="center", color="w", fontsize=10, fontweight="bold")
ax[0].set_xticks(x); ax[0].set_xticklabels(cfg, fontsize=8); ax[0].set_ylabel("QPS (T=64, L=100)")
ax[0].set_title("(A) throughput — cumulative"); ax[0].grid(axis="y", alpha=0.3)

# B: remote% (PQ filter's lever)
b = ax[1].bar(x, remote, color=COL, edgecolor="k", width=0.62)
for i, v in enumerate(remote):
    ax[1].text(i, v+1.5, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
ax[1].set_xticks(x); ax[1].set_xticklabels(cfg, fontsize=8); ax[1].set_ylabel("remote vector access %")
ax[1].set_title("(B) PQ filter cuts remote-fetch COUNT\n(batched does NOT change it)"); ax[1].grid(axis="y", alpha=0.3)

# C: us/call (batched's lever)
b = ax[2].bar(x, uscall, color=COL, edgecolor="k", width=0.62)
for i, v in enumerate(uscall):
    ax[2].text(i, v+0.4, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")
ax[2].set_xticks(x); ax[2].set_xticklabels(cfg, fontsize=8); ax[2].set_ylabel("us / call (per-fetch latency)")
ax[2].set_title("(C) batched fetch hides LATENCY\n(PQ also lowers it via fewer fetches)"); ax[2].grid(axis="y", alpha=0.3)

fig.suptitle("Q4b / M10: CAP ablation (deep100M 4p) — PQ filter = fetch-COUNT reducer, "
             "batched fetch = LATENCY hider; distinct & largely additive (full = 2.4× rr)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
ts = time.strftime("%Y%m%d_%H%M%S")
out = f"/experiments/results/deep100M/q4_cap_ablation/{ts}_q4b_cap_ablation.png"
fig.savefig(out, dpi=130)
print(f"wrote {out}")
