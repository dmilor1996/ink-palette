"""Assemble the site: tiles -> img/*.jpg, palette data -> index.html.
Usage: build.py <tiles_dir> <template> <dist_dir> [swatch-cuts.json]
Tiles that are already JPEG (a rebuild from img/) are copied as they are: re-encoding them on every build would degrade them."""
import sys, os, json, shutil
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as P

TILES, TEMPLATE, DIST = sys.argv[1], sys.argv[2], sys.argv[3]
CUTS = sys.argv[4] if len(sys.argv) > 4 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'swatch-cuts.json')
from detect import PHOTOS
PHOTO_FILES = ['photos/palette-%d.jpg' % (i + 1) for i in range(len(PHOTOS))]
cuts = json.load(open(CUTS))
photo_index = {}
for i, (fn, cols) in enumerate(PHOTOS):
    photo_index[fn] = i; photo_index[PHOTO_FILES[i]] = i
os.makedirs(os.path.join(DIST, 'img'), exist_ok=True)

def tile_file(code):
    for ext in ('.png', '.jpg'):
        f = os.path.join(TILES, code + ext)
        if os.path.exists(f):
            return f
    raise FileNotFoundError(code)

def tile(code):
    return cv2.imread(tile_file(code))

def lab_of(code):
    """CIELAB (D65) of the tile's average light"""
    im = tile(code).astype(np.float64) / 255.0
    lin = np.where(im <= 0.04045, im / 12.92, ((im + 0.055) / 1.055) ** 2.4).reshape(-1, 3).mean(0)[::-1]   # RGB
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    x, y, z = (M @ lin) / np.array([0.95047, 1.0, 1.08883])
    f = lambda t: t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    return [round(116 * f(y) - 16, 1), round(500 * (f(x) - f(y)), 1), round(200 * (f(y) - f(z)), 1)]

def ph_of(code):
    """[photo index, left%, top%, width%, height%] of the swatch on its original photo"""
    v = cuts[code]
    q = np.array(v['quad']); W, H = 1659.0, 2212.0
    x0, y0 = q[:, 0].min(), q[:, 1].min(); x1, y1 = q[:, 0].max(), q[:, 1].max()
    return [photo_index[v['photo']], round(100 * x0 / W, 2), round(100 * y0 / H, 2), round(100 * (x1 - x0) / W, 2), round(100 * (y1 - y0) / H, 2)]

def hexof(code):
    im = tile(code)
    b, g, r = [int(round(v)) for v in im.reshape(-1, 3).mean(0)]
    return '#%02x%02x%02x' % (r, g, b)

groups = []
for gid in P.ORDER:
    sec = gid in P.SEC
    groups.append({
        'id': gid, 'name': P.SEC[gid][0] if sec else P.BASE[gid], 'type': 's' if sec else 'b',
        'mix': [[b, p] for b, p in P.SEC[gid][1].items()] if sec else None,
        'hex': hexof(gid + {'INK50P': '15', 'INK80P': '15', 'INK90P': '40', 'INKBP': '40'}.get(gid, '5')),
    })

shades, total = [], 0
for r in P.rows():
    code = r['code']
    gid = next(g for g in sorted(P.ORDER, key=len, reverse=True) if code.startswith(g) and (code[len(g):] in ('40', '30', '20', '15', '10', '5', '2', 'B1005', 'B1002')))
    lv = code[len(gid):]
    if lv.startswith('B'):
        own = {'B1005': 5.0, 'B1002': 2.0}[lv]
        comp = [['INKBP', 100.0]] + [[b, round(v, 4)] for b, v in P.parts(gid, own).items()]
    else:
        own = float(lv) * 10
        comp = [[b, round(v, 4)] for b, v in P.parts(gid, own).items()]
    assert sum(v for _, v in comp) < 1000
    shades.append({'c': code, 'g': gid, 'lv': lv, 'tone': r['tone'], 'names': r['names'], 'comp': comp, 'own': own, 'lab': lab_of(code), 'ph': ph_of(code)})
    im = tile(code)
    assert im is not None and im.shape[:2] == (300, 240), code
    src, out = tile_file(code), os.path.join(DIST, 'img', code + '.jpg')
    if not src.endswith('.jpg'):
        cv2.imwrite(out, im, [cv2.IMWRITE_JPEG_QUALITY, 90])
    elif not (os.path.exists(out) and os.path.samefile(src, out)):
        shutil.copyfile(src, out)
    total += os.path.getsize(out)

Ls = [s['lab'][0] for s in shades]; Cs = [float(np.hypot(s['lab'][1], s['lab'][2])) for s in shades]
rng = {'L': [float(np.floor(min(Ls)) - 1), float(np.ceil(max(Ls)) + 1)], 'C': float(np.ceil(max(Cs)) + 2)}
print('L range', rng['L'], 'max chroma', round(max(Cs), 1))
photos = [{'file': PHOTO_FILES[i], 'w': 1659, 'h': 2212, 'groups': cols} for i, (fn, cols) in enumerate(PHOTOS)]
data = json.dumps({'groups': groups, 'shades': shades, 'range': rng, 'photos': photos}, ensure_ascii=False, separators=(',', ':'))
html = open(TEMPLATE, encoding='utf-8').read()
assert '/*__DATA__*/null' in html
open(os.path.join(DIST, 'index.html'), 'w', encoding='utf-8').write(html.replace('/*__DATA__*/null', data))
print('groups', len(groups), 'shades', len(shades), 'images %.2f MB' % (total / 1e6), 'html %d KB' % (len(html.encode()) // 1024 + len(data.encode()) // 1024))
print([(g['id'], g['hex']) for g in groups[:10]])
