"""Measure the paper colour around every swatch, fit a smooth lighting field per photo,
and report how much the five photos differ. Usage: wb.py <uploads> <out_dir> [apply <tiles_in> <tiles_out>]"""
import sys, os, json
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect import PHOTOS
SRC, OUT = sys.argv[1], sys.argv[2]
rep = json.load(open(os.path.join(OUT, 'report.json')))

def to_lin(u8):
    c = u8.astype(np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
def to_srgb(lin):
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)

def paper_near(lin, quad):
    H, W = lin.shape[:2]
    q = np.array(quad)
    l, r, t, b = q[:, 0].min(), q[:, 0].max(), q[:, 1].min(), q[:, 1].max()
    bands = [(t - 22, t - 9, l + 20, r - 20), (t + 20, b - 20, l - 22, l - 9), (t + 20, b - 20, r + 9, r + 22)]
    px = []
    for y0, y1, x0, x1 in bands:
        y0, y1, x0, x1 = [int(round(v)) for v in (max(0, y0), min(H, y1), max(0, x0), min(W, x1))]
        if y1 - y0 > 4 and x1 - x0 > 4:
            px.append(lin[y0:y1, x0:x1].reshape(-1, 3))
    px = np.vstack(px)
    Y = px @ np.array([0.0722, 0.7152, 0.2126])          # BGR order
    lo, hi = np.percentile(Y, [55, 95])                    # skip shadows and specks
    return px[(Y >= lo) & (Y <= hi)].mean(0)

fields, allp = {}, []
for fn, colors in PHOTOS:
    lin = to_lin(cv2.imread(os.path.join(SRC, fn)))
    pts, vals = [], []
    for code, v in rep.items():
        if v['photo'] != fn: continue
        q = np.array(v['quad']); pts.append(q.mean(0)); vals.append(paper_near(lin, v['quad']))
    pts, vals = np.array(pts), np.array(vals)
    x, y = pts[:, 0] / 1659.0, pts[:, 1] / 2212.0
    A = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], 1)
    coef, *_ = np.linalg.lstsq(A, vals, rcond=None)
    fit = A @ coef
    fields[fn] = coef.tolist()
    allp.append(vals)
    m = vals.mean(0)[::-1]
    lab = cv2.cvtColor((to_srgb(vals.mean(0))[None, None, :] * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)[0, 0].astype(int)
    print(fn, 'paper sRGB', (to_srgb(m) * 255).round().astype(int), 'L*a*b*', lab[0] * 100 // 255, lab[1] - 128, lab[2] - 128,
          '| brightness across the photo: %.0f%%..%.0f%% of its mean' % (100 * fit.sum(1).min() / fit.sum(1).mean(), 100 * fit.sum(1).max() / fit.sum(1).mean()),
          '| fit error %.1f%%' % (100 * np.abs(fit - vals).mean() / vals.mean()))
allp = np.vstack(allp)
target = float(np.median(allp @ np.array([0.0722, 0.7152, 0.2126])))
print('target paper luminance %.3f -> sRGB %d' % (target, round(float(to_srgb(np.array(target))) * 255)))
json.dump({'fields': fields, 'target': target}, open(os.path.join(OUT, 'wb.json'), 'w'))

if len(sys.argv) > 3 and sys.argv[3] == 'apply':
    TIN, TOUT = sys.argv[4], sys.argv[5]
    os.makedirs(TOUT, exist_ok=True)
    for code, v in rep.items():
        cx, cy = np.array(v['quad']).mean(0)
        x, y = cx / 1659.0, cy / 2212.0
        paper = np.array([1, x, y, x * x, x * y, y * y]) @ np.array(fields[v['photo']])
        gain = target / paper                                   # paper becomes the same neutral grey everywhere
        tile = to_lin(cv2.imread(os.path.join(TIN, code + '.png')))
        cv2.imwrite(os.path.join(TOUT, code + '.png'), (to_srgb(tile * gain) * 255 + 0.5).astype(np.uint8))
    print('corrected tiles written')
