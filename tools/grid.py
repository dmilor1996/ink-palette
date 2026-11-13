"""Multi-threshold swatch detection -> 6x5 grid of boxes per photo.
Usage: grid.py <uploads_dir> <out_dir>"""
import sys, json, os
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect import PHOTOS, mask_for

SRC, OUT = sys.argv[1], sys.argv[2]


def boxes_at(d, thr):
    m = (d > thr).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cs0, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = np.zeros_like(m)
    cv2.drawContours(m, cs0, -1, 255, -1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((45, 45), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        x, y, w, h = cv2.boundingRect(c)
        if 190 < w < 290 and 240 < h < 350 and cv2.contourArea(c) > 0.9 * w * h:
            out.append((x, y, w, h))
    return out


def clusters(vals, k):
    """split sorted 1-D values into k groups at the k-1 widest gaps"""
    v = np.sort(np.array(vals))
    gaps = np.diff(v)
    cut = np.sort(np.argsort(gaps)[-(k - 1):])
    edges = [(v[i] + v[i + 1]) / 2 for i in cut]
    return edges


def run(pi, fn):
    img = cv2.imread(os.path.join(SRC, fn))
    d = mask_for(img)
    found = {}  # thr -> boxes
    allb = []
    for thr in range(4, 40, 2):
        for b in boxes_at(d, thr):
            allb.append((thr,) + b)
    cx = [b[1] + b[3] / 2 for b in allb]
    cy = [b[2] + b[4] / 2 for b in allb]
    ex, ey = clusters(cx, 6), clusters(cy, 5)
    cells = {}
    for thr, x, y, w, h in allb:
        c = int(np.searchsorted(ex, x + w / 2)); r = int(np.searchsorted(ey, y + h / 2))
        cells.setdefault((r, c), []).append((thr, x, y, w, h))
    grid = {}
    for (r, c), lst in cells.items():
        lst.sort()
        # the box is stable over a range of thresholds; take the median of the lowest few
        take = np.array([b[1:] for b in lst[:3]], dtype=float)
        grid[(r, c)] = [float(v) for v in np.median(take, axis=0)] + [lst[0][0], len(lst)]
    return img, grid


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for pi, (fn, colors) in enumerate(PHOTOS):
        img, grid = run(pi, fn)
        vis = img.copy()
        miss = [(r, c) for r in range(5) for c in range(6) if (r, c) not in grid]
        for (r, c), (x, y, w, h, t, n) in grid.items():
            cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 3)
            cv2.putText(vis, 't%d n%d' % (t, n), (int(x) + 8, int(y) + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        print(fn, 'cells', len(grid), 'missing', miss)
        cv2.imwrite(os.path.join(OUT, 'grid%d.jpg' % pi), cv2.resize(vis, None, fx=0.5, fy=0.5))
        res[fn] = {'%d,%d' % k: v for k, v in grid.items()}
    json.dump(res, open(os.path.join(OUT, 'grid.json'), 'w'), indent=1)
