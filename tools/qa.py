"""QA of cut tiles: does the outer ring differ from the ring just inside it (paper or shadow left in)?
Also builds contact sheets. Usage: qa.py <tiles_dir> <out_dir>"""
import sys, os, json
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect import PHOTOS
T, OUT = sys.argv[1], sys.argv[2]
LEVELS = ['40', '15', '5', '2', 'B1005', 'B1002']; WLEVELS = ['40', '30', '20', '15', '10', '5']
flags = []
for pi, (fn, colors) in enumerate(PHOTOS):
    rows = []
    for col in colors:
        lv = WLEVELS if col == 'INKBP' else LEVELS
        row = []
        for l in lv:
            code = col + l
            im = cv2.imread(os.path.join(T, code + '.png'))
            lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
            H, W = lab.shape[:2]
            worst = 0; side = ''
            for name, outer, inner in (
                ('T', lab[0:3, 20:W-20], lab[8:20, 20:W-20]), ('B', lab[H-3:H, 20:W-20], lab[H-20:H-8, 20:W-20]),
                ('L', lab[20:H-20, 0:3], lab[20:H-20, 8:20]), ('R', lab[20:H-20, W-3:W], lab[20:H-20, W-20:W-8])):
                de = float(np.linalg.norm(outer.reshape(-1, 3).mean(0) - inner.reshape(-1, 3).mean(0)))
                if de > worst: worst, side = de, name
            if worst > 6: flags.append((code, side, round(worst, 1)))
            pad = cv2.copyMakeBorder(im, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=(255, 0, 255))
            row.append(pad)
        rows.append(np.hstack(row))
    cv2.imwrite(os.path.join(OUT, 'sheet%d.jpg' % pi), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
print('flagged', len(flags)); print(flags)
