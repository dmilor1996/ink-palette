"""Fit a quadrilateral to every swatch and cut it out as a straightened tile.
Usage: refine.py <uploads_dir> <out_dir> <tiles_dir>"""
import sys, json, os
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect import PHOTOS

SRC, OUT, TILES = sys.argv[1], sys.argv[2], sys.argv[3]
LEVELS = ['40', '15', '5', '2', 'B1005', 'B1002']
WLEVELS = ['40', '30', '20', '15', '10', '5']
TW, TH = 240, 300
INSET = 7.0
# the two whitest swatches have almost no contrast with the paper: borders read off brightness profiles (left, right, top, bottom)
MANUAL = {'INKBP40': (25.5, 262.0, 1781.0, 2064.0), 'INKBP30': (298.5, 534.0, 1777.0, 2060.0)}


def dmap(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    Ls = cv2.GaussianBlur(L, (0, 0), 3)
    bg = cv2.dilate(Ls, cv2.getStructuringElement(cv2.MORPH_RECT, (181, 181)))
    bg = cv2.GaussianBlur(bg, (0, 0), 60)
    paper = (bg - Ls) < 5
    a0, b0 = np.median(A[paper]), np.median(B[paper])
    g = lambda c: cv2.GaussianBlur(c, (0, 0), 1.3)
    d = np.sqrt((bg - g(L)) ** 2 + (g(A) - a0) ** 2 + (g(B) - b0) ** 2)
    grad_x = np.abs(cv2.Sobel(g(L), cv2.CV_32F, 1, 0, ksize=3))
    grad_y = np.abs(cv2.Sobel(g(L), cv2.CV_32F, 0, 1, ksize=3))
    return d, grad_x, grad_y


def first_sustained(profile, t, run=5):
    """index of first sample from which `run` samples in a row exceed t"""
    above = (profile > t).astype(np.int32)
    k = np.convolve(above, np.ones(run, dtype=np.int32), 'valid')
    idx = np.nonzero(k == run)[0]
    return int(idx[0]) if len(idx) else None


def robust_line(us, vs):
    """fit v = a*u + b, dropping outliers twice"""
    us, vs = np.array(us, float), np.array(vs, float)
    keep = np.ones(len(us), bool)
    a, b = 0.0, float(np.median(vs))
    for _ in range(3):
        if keep.sum() < 6:
            break
        a, b = np.polyfit(us[keep], vs[keep], 1)
        res = np.abs(vs - (a * us + b))
        keep = res < max(1.5, 2.0 * np.median(res[keep]))
    return a, b


def edges_by_d(d, box):
    """scan from inside the swatch outwards until the paper starts (so labels and neighbours never interfere)"""
    x, y, w, h = [int(round(v)) for v in box]
    core = d[y + h // 4: y + 3 * h // 4, x + w // 4: x + 3 * w // 4]
    din = float(np.median(core))
    t = float(np.clip(0.5 * din, 3.5, 40))
    IN, OUTM = 50, 18
    H, W = d.shape
    pts = {'L': ([], []), 'R': ([], []), 'T': ([], []), 'B': ([], [])}

    def leave(profile):  # profile runs inside -> outside; index of the last sample still on the swatch
        i = first_sustained(-profile, -t, run=4)
        return None if i is None else i - 1

    for yy in range(y + int(0.15 * h), y + int(0.85 * h), 3):
        xa, xb = max(0, x - OUTM), x + IN
        i = leave(d[yy, xa:xb][::-1])
        if i is not None and i >= 0: pts['L'][0].append(yy); pts['L'][1].append(xb - 1 - i)
        xa, xb = x + w - IN, min(W, x + w + OUTM)
        i = leave(d[yy, xa:xb])
        if i is not None and i >= 0: pts['R'][0].append(yy); pts['R'][1].append(xa + i)
    for xx in range(x + int(0.15 * w), x + int(0.85 * w), 3):
        ya, yb = max(0, y - OUTM), y + IN
        i = leave(d[ya:yb, xx][::-1])
        if i is not None and i >= 0: pts['T'][0].append(xx); pts['T'][1].append(yb - 1 - i)
        ya, yb = y + h - IN, min(H, y + h + OUTM)
        i = leave(d[ya:yb, xx])
        if i is not None and i >= 0: pts['B'][0].append(xx); pts['B'][1].append(ya + i)
    return {k: robust_line(*v) for k, v in pts.items()}, din, t


def hough_profile(g, pos, lo, hi, vertical, win, slope0):
    """for each candidate offset near `pos`: strength of the best near-straight line there, and its slope"""
    c = (lo + hi) / 2.0
    span = np.arange(lo, hi)
    ps = np.arange(int(pos) - win, int(pos) + win + 1)
    S = np.zeros(len(ps)); SL = np.zeros(len(ps))
    for s in slope0 + np.linspace(-0.02, 0.02, 9):
        off = s * (span - c)
        for i, p in enumerate(ps):
            q = np.round(p + off).astype(int)
            v = g[span, q].mean() if vertical else g[q, span].mean()
            if v > S[i]:
                S[i], SL[i] = v, s
    return ps, S, SL, c


def best_pair(pa, Sa, pb, Sb, size, tol=5):
    """two parallel sides whose distance matches the known swatch size"""
    best = (-1, 0, 0)
    for i, p in enumerate(pa):
        for j, q in enumerate(pb):
            if abs((q - p) - size) <= tol and Sa[i] + Sb[j] > best[0]:
                best = (Sa[i] + Sb[j], i, j)
    return best[1], best[2]


def edges_by_grad(gx, gy, box, slopes, size, win=18):
    """white-on-white swatches: snap the sides to the strongest straight lines near the guess,
    keeping the swatch the same size as its neighbours"""
    x, y, w, h = [int(round(v)) for v in box]
    ylo, yhi = y + int(0.12 * h), y + int(0.88 * h)
    xlo, xhi = x + int(0.12 * w), x + int(0.88 * w)
    out = {}
    pl, Sl, sl, cy = hough_profile(gx, x, ylo, yhi, True, win, slopes['L'])
    pr, Sr, sr, _ = hough_profile(gx, x + w, ylo, yhi, True, win, slopes['R'])
    i, j = best_pair(pl, Sl, pr, Sr, size[0])
    out['L'] = (sl[i], pl[i] - sl[i] * cy); out['R'] = (sr[j], pr[j] - sr[j] * cy)
    pt, St, st, cx = hough_profile(gy, y, xlo, xhi, False, win, slopes['T'])
    pb, Sb, sb, _ = hough_profile(gy, y + h, xlo, xhi, False, win, slopes['B'])
    i2, j2 = best_pair(pt, St, pb, Sb, size[1])
    out['T'] = (st[i2], pt[i2] - st[i2] * cx); out['B'] = (sb[j2], pb[j2] - sb[j2] * cx)
    return out, {'L': Sl[i], 'R': Sr[j], 'T': St[i2], 'B': Sb[j2]}


def quad_size(lines):
    q = quad(lines, 0)
    return float(np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2, float(np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2


def quad(lines, inset):
    """corners (TL, TR, BR, BL). inset: one number or per-side dict, in source pixels"""
    ins = inset if isinstance(inset, dict) else {k: inset for k in 'LRTB'}
    (aL, bL), (aR, bR), (aT, bT), (aB, bB) = lines['L'], lines['R'], lines['T'], lines['B']
    bL, bR, bT, bB = bL + ins['L'], bR - ins['R'], bT + ins['T'], bB - ins['B']

    def cross(av, bv, ah, bh):  # x = av*y + bv ; y = ah*x + bh
        xx = (av * bh + bv) / (1 - av * ah)
        return [xx, ah * xx + bh]
    return np.array([cross(aL, bL, aT, bT), cross(aR, bR, aT, bT), cross(aR, bR, aB, bB), cross(aL, bL, aB, bB)], np.float32)


def warp(img, q):
    M = cv2.getPerspectiveTransform(q, np.array([[0, 0], [TW, 0], [TW, TH], [0, TH]], np.float32))
    return cv2.warpPerspective(img, M, (TW, TH), flags=cv2.INTER_CUBIC)


def edge_excess(tile):
    """per side: how many pixels at the very edge still look unlike the swatch just inside (paper, shadow)"""
    lab = cv2.cvtColor(tile, cv2.COLOR_BGR2LAB).astype(np.float32)
    H, W = lab.shape[:2]
    strips = {
        'T': lab[:26, int(.15 * W):int(.85 * W)].mean(1),
        'B': lab[::-1][:26, int(.15 * W):int(.85 * W)].mean(1),
        'L': lab[int(.15 * H):int(.85 * H), :26].mean(0),
        'R': lab[int(.15 * H):int(.85 * H), ::-1][:, :26].mean(0),
    }
    out = {}
    for k, P in strips.items():
        ref = np.median(P[12:], axis=0)
        dev = np.linalg.norm(P - ref, axis=1)
        across = k in 'TB'                      # top/bottom strips run across the grain: a clean signal
        thr = max(4.0 if across else 6.0, (3.0 if across else 4.0) * float(np.median(dev[12:])) + 1.0)
        run = 0                                 # paper/shadow always starts at the very edge
        while run < (12 if across else 6) and dev[run] > thr:
            run += 1
        out[k] = run
    return out


def main():
    os.makedirs(OUT, exist_ok=True); os.makedirs(TILES, exist_ok=True)
    grid = json.load(open(os.path.join(OUT, 'grid.json')))
    report = {}
    for pi, (fn, colors) in enumerate(PHOTOS):
        img = cv2.imread(os.path.join(SRC, fn))
        d, gx, gy = dmap(img)
        g = {tuple(map(int, k.split(','))): v for k, v in grid[fn].items()}
        vis = img.copy()
        fitted = {}
        for r in range(5):
            for c in range(6):
                levels = WLEVELS if colors[r] == 'INKBP' else LEVELS
                code = colors[r] + levels[c]
                method = 'd'
                if (r, c) in g:
                    box = g[(r, c)][:4]
                    lines, din, t = edges_by_d(d, box)
                    fitted[(r, c)] = lines
                else:
                    # guess from neighbours: x/w from the swatch above (plus the drift between the two rows above it),
                    # y/h from the detected swatch in the same row, tilted like the row above
                    up, up2 = g[(r - 1, c)], g[(r - 2, c)]
                    ref = g[(r, 5)]; refup = g[(r - 1, 5)]
                    bx = up[0] + (up[0] - up2[0]); bw = up[2] + (up[2] - up2[2])
                    by = ref[1] + (up[1] - refup[1]); bh = ref[3] + (up[3] - refup[3])
                    box = [bx, by, bw, bh]
                    la, lr = fitted[(r - 1, c)], edges_by_d(d, g[(r, 5)][:4])[0]
                    slopes = {'L': la['L'][0], 'R': la['R'][0], 'T': lr['T'][0], 'B': lr['B'][0]}
                    # same physical swatch: size of the one above, scaled by how the detected swatch of this row
                    # compares with the one above it
                    wu, hu = quad_size(la); w5, h5 = quad_size(lr); w5u, h5u = quad_size(fitted[(r - 1, 5)])
                    size = (wu * w5 / w5u, hu * h5 / h5u)
                    lines, strength = edges_by_grad(gx, gy, box, slopes, size)
                    print(code, 'guess', [round(v) for v in box], 'size', [round(v) for v in size], 'strength', {k: round(v, 1) for k, v in strength.items()})
                    method, din, t = 'grad', 0, 0
                if code in MANUAL:
                    ml, mr, mt, mb = MANUAL[code]
                    lines = {'L': (0.0, ml), 'R': (0.0, mr), 'T': (0.0, mt), 'B': (0.0, mb)}
                    method = 'manual'
                q0 = quad(lines, 0)
                ins = {k: INSET for k in 'LRTB'}
                for _ in range(3):   # trim any side that still shows paper or shadow
                    q = quad(lines, ins)
                    tile = warp(img, q)
                    ex = edge_excess(tile)
                    if not any(ex.values()):
                        break
                    for k, v in ex.items():
                        ins[k] = min(ins[k] + v, 22 if k in 'TB' else 15)
                cv2.imwrite(os.path.join(TILES, code + '.png'), tile)
                cv2.polylines(vis, [q0.astype(np.int32)], True, (0, 255, 0) if method == 'd' else (255, 0, 255), 1)
                cv2.polylines(vis, [q.astype(np.int32)], True, (0, 0, 255), 1)
                wq = float(np.linalg.norm(q0[1] - q0[0])); hq = float(np.linalg.norm(q0[3] - q0[0]))
                report[code] = {'photo': fn, 'method': method, 'din': round(din, 1), 't': round(t, 1),
                                'w': round(wq, 1), 'h': round(hq, 1), 'ins': {k: round(v) for k, v in ins.items()}, 'left': ex, 'quad': q0.round(1).tolist(), 'cut': q.round(1).tolist()}
        cv2.imwrite(os.path.join(OUT, 'quad%d.jpg' % pi), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
    json.dump(report, open(os.path.join(OUT, 'report.json'), 'w'), indent=1)
    ws = np.array([v['w'] for v in report.values()]); hs = np.array([v['h'] for v in report.values()])
    print('tiles', len(report), 'w %.0f..%.0f  h %.0f..%.0f' % (ws.min(), ws.max(), hs.min(), hs.max()))
    for fn, _ in PHOTOS:
        sel = [v for v in report.values() if v['photo'] == fn]
        w = np.array([v['w'] for v in sel]); h = np.array([v['h'] for v in sel])
        odd = [k for k, v in report.items() if v['photo'] == fn and (abs(v['w'] - np.median(w)) > 8 or abs(v['h'] - np.median(h)) > 8)]
        print(fn, 'median w %.0f h %.0f' % (np.median(w), np.median(h)), 'odd size:', odd)


main()
