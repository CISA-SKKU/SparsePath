import numpy as np
for ds in ["deep100M","spacev100M","sift100M"]:
    p=f"/experiments/artifacts/{ds}/graph/checkpoints/lits_k4_iter150.tags"
    with open(p,"rb") as f:
        n,d=np.fromfile(f,np.uint32,2)
        t=np.fromfile(f,np.uint32,int(n)*int(d))
    nz=t[t!=0]; g=nz.astype(np.int64)-1
    inw=((g>=150_000_000)&(g<250_000_000)).sum()
    print(f"{ds:11s} tags={n} nonzero={len(nz)} gid[min,max]=[{g.min()},{g.max()}] "
          f"in[150M,250M)={inw} ({100.0*inw/len(nz):.4f}%) OUTSIDE={len(nz)-inw}", flush=True)
