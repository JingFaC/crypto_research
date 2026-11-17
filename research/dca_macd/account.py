import heapq
import numpy as np
def acct(cs, start, T, N=10, DIV=2.0, cap=None):
    heap = []; realized = 0.0; path = np.zeros(T + 2); step = np.zeros(T + 2); nb = 0; opn = {}; skipped = 0; used = 0
    for k, (i0, i1, u, bust) in enumerate(cs):
        if i0 < start: continue
        while heap and heap[0][0] < i0:
            _, j = heapq.heappop(heap); realized += opn[j][4] * (float(opn[j][2][-1]) - 1.0)
        if cap is not None and len(heap) >= cap: skipped += 1; continue
        Eprev = 1.0 + realized
        for _, j in heap:
            a, _, uj, _, sj = opn[j]
            if a < i0: Eprev += sj * (float(uj[min(i0 - 1 - a, len(uj) - 1)]) - 1.0)
        s_ = Eprev / (N * DIV); opn[k] = (i0, i1, u, bust, s_); heapq.heappush(heap, (i1, k)); nb += bust; used += 1
        path[i0:i1 + 1] += s_ * (u.astype(np.float64) - 1.0); step[i1 + 1] += s_ * (float(u[-1]) - 1.0)
    Et = 1.0 + path[:T] + np.cumsum(step)[:T]
    return Et, dict(nbust=int(nb), used=used, skipped=skipped)
