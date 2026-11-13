"""Find the 30 swatches on each palette photo. Usage: detect.py <uploads_dir> <out_dir>"""
import sys, json, os
import numpy as np, cv2

SRC, OUT = sys.argv[1], sys.argv[2]
THR = 7
PHOTOS = [
    ('2f9bba93-image.jpg', ['INK10P', 'INK11P', 'INK20P', 'INK30P', 'INK40P']),
    ('e928a57e-image.jpg', ['INK50P', 'INK60P', 'INK80P', 'INK90P', 'INKBP']),
    ('eb9e5382-image.jpg', ['INK12S', 'INK23S', 'INK44S', 'INK47S', 'INK55S']),
    ('f3b7b42a-image.jpg', ['INK57S', 'INK61S', 'INK65S', 'INK66S', 'INK67S']),
    ('08d23473-image.jpg', ['INK81S', 'INK82S', 'INK83S', 'INK86S', 'INK88S']),
]


def mask_for(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    # paper brightness field: brightest thing in a wide neighbourhood, smoothed
    Ls = cv2.GaussianBlur(L, (0, 0), 3)
    bg = cv2.dilate(Ls, cv2.getStructuringElement(cv2.MORPH_RECT, (181, 181)))
    bg = cv2.GaussianBlur(bg, (0, 0), 60)
    paper = (bg - Ls) < 5
    a0, b0 = np.median(A[paper]), np.median(B[paper])
    d = np.sqrt((bg - Ls) ** 2 + (cv2.GaussianBlur(A, (0, 0), 3) - a0) ** 2 + (cv2.GaussianBlur(B, (0, 0), 3) - b0) ** 2)
    return d


def detect(img, thr):
    d = mask_for(img)
    m = (d > thr).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cs0, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = np.zeros_like(m)
    cv2.drawContours(m, cs0, -1, 255, -1)                                  # fill holes
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((45, 45), np.uint8))   # kills label text
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cs:
        x, y, w, h = cv2.boundingRect(c)
        if 150 < w < 330 and 200 < h < 400 and cv2.contourArea(c) > 0.8 * w * h:
            boxes.append((x, y, w, h, c))
    return boxes, m


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for pi, (fn, colors) in enumerate(PHOTOS):
        img = cv2.imread(os.path.join(SRC, fn))
        for thr in (THR,):
            boxes, m = detect(img, thr)
            vis = img.copy()
            for x, y, w, h, c in boxes:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)
            print(fn, 'thr', thr, 'found', len(boxes))
            cv2.imwrite(os.path.join(OUT, 'det%d.jpg' % pi), cv2.resize(vis, None, fx=0.5, fy=0.5))
            cv2.imwrite(os.path.join(OUT, 'mask%d.png' % pi), cv2.resize(m, None, fx=0.25, fy=0.25))
