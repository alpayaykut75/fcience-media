"""Fcience standart kapanış kartı: kil şişe maskot, stop-motion el sallama + göz kırpma (poz değiştirme, bükme yok)."""
import math, random, numpy as np, cv2, cairo

D = '/home/claude/end/'
def _ld(f): return cv2.cvtColor(cv2.imread(D + f), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
A, B, C = _ld('endA.png'), _ld('endB.png'), _ld('endC.png')
EH, EW = A.shape[:2]

def _mask(X, box, thr=0.09, dil=19, blur=9):
    d = np.abs(A - X).mean(2); m = np.zeros_like(d)
    x0, x1, y0, y1 = box
    m[y0:y1, x0:x1] = (d[y0:y1, x0:x1] > thr)
    m = cv2.dilate(m, np.ones((dil, dil))); m = cv2.GaussianBlur(m, (0, 0), blur)
    return np.clip(m * 1.5, 0, 1)[..., None]
MH = _mask(B, (90, 380, 640, 980))
ME = _mask(C, (320, 630, 780, 970))
DUR = 2.8
STOP = 12
W, H = 1080, 1920

def background(tl):
    """tl: seconds since card start. returns float RGB WxH image and affine M (image->screen)."""
    f = int(tl * STOP)
    img = A
    if (f // 3) % 2 == 1:
        img = img * (1 - MH) + B * MH
    if f in (17, 18):
        img = img * (1 - ME) + C * ME
    # lamp flicker
    r = random.Random(f)
    g = np.exp(-(((np.arange(EW)[None, :] - 112) / 90) ** 2 + ((np.arange(EH)[:, None] - 805) / 90) ** 2) / 2)[..., None]
    img = img + g * np.array([1.0, 0.75, 0.4], np.float32) * (0.10 + 0.04 * r.random())
    z = 1.0 + 0.035 * min(1, tl / DUR)
    s = W / EW * z
    jx, jy = r.uniform(-1, 1), r.uniform(-1, 1)
    cx, cy = 470, 900
    M = np.float32([[s, 0, W / 2 - cx * s + jx], [0, s, H / 2 - cy * s + jy + 30]])
    out = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out, M

_rng = np.random.default_rng(3)
_BUB = [(_rng.uniform(-35, 35), _rng.uniform(0, 1), _rng.uniform(6, 13), _rng.uniform(0.7, 1.3)) for _ in range(7)]

def _back(x, s=1.7): x -= 1; return x * x * ((s + 1) * x + s) + 1

def overlay(ctx, tl, M, label_fn):
    f = int(tl * STOP); tq = f / STOP
    for (ox, ph, rr, sp) in _BUB:
        k = (ph + tq * 0.55 * sp) % 1
        x = 462 + ox + 14 * math.sin(tq * 3 + ph * 6); y = 610 - k * 300
        sx, sy = M[0, 0] * x + M[0, 2], M[1, 1] * y + M[1, 2]
        ctx.new_path(); ctx.arc(sx, sy, rr * (0.6 + 0.7 * k), 0, 6.283)
        ctx.set_source_rgba(0.62, 0.95, 0.25, 0.75 * (1 - k)); ctx.fill_preserve()
        ctx.set_source_rgba(0.25, 0.45, 0.1, 0.6 * (1 - k)); ctx.set_line_width(2); ctx.stroke()
    a = _back(min(1, max(0, (tq - 0.1) / 0.3))) if tq < 0.4 else 1
    ctx.save(); ctx.translate(540, 240); ctx.scale(a, a)
    ctx.select_font_face('Archivo Black', 0, 1); ctx.set_font_size(112)
    e = ctx.text_extents("FCIENCE")
    ctx.move_to(-e.x_advance / 2 + 5, 7); ctx.text_path("FCIENCE"); ctx.set_source_rgba(0, 0, 0, 0.35); ctx.fill()
    ctx.move_to(-e.x_advance / 2, 0); ctx.text_path("FCIENCE"); ctx.set_source_rgba(0.12, 0.24, 0.08, 1); ctx.set_line_width(16); ctx.set_line_join(1); ctx.stroke()
    ctx.move_to(-e.x_advance / 2, 0); ctx.set_source_rgba(0.98, 0.96, 0.88, 1); ctx.show_text("FCIENCE"); ctx.new_path()
    ctx.restore()
    label_fn(["Takip et,", "tuhaf bilimi kaçırma!"], 0.55)
