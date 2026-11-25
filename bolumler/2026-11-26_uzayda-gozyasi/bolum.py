"""Fcience – Uzayda gözyaşı kil videosu."""
import sys, os, math, random, subprocess, numpy as np, cv2, cairo
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _ROOT = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path.insert(0, _ROOT + "/lib"); _sys.path.insert(0, _ROOT + "/kapanis")
import endcard
from fcience import label, bubble, surf_from, F, back, prog, ease, clamp, glow_dot, VIG, W, H, FPS, STOP

END = 27.2
def ld(f): return cv2.cvtColor(cv2.imread(f'{_HERE}/{f}'), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
A1 = ld('a1.png'); H1, W1 = A1.shape[:2]
A2 = cv2.resize(ld('a2.jpg'), (W1, H1), interpolation=cv2.INTER_AREA)
A3 = ld('a3.jpg'); A4 = ld('a4.jpg')
GRID = {}
def grid(img):
    k = img.shape[:2]
    if k not in GRID: GRID[k] = np.mgrid[0:k[0], 0:k[1]].astype(np.float32)
    return GRID[k]
def gauss(img, cx, cy, sx, sy=None):
    yy, xx = grid(img); sy = sy or sx
    return np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2)
def warp(img, dx, dy):
    yy, xx = grid(img); return cv2.remap(img, xx - dx, yy - dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
def add_glow(img, cx, cy, r, col, amt): return img + gauss(img, cx, cy, r)[..., None] * np.array(col, np.float32) * amt

d = np.abs(A1 - A2).mean(2); EYE = np.zeros_like(d); EYE[440:600, 290:480] = d[440:600, 290:480] > 0.07
EYE = np.clip(cv2.GaussianBlur(cv2.dilate(EYE, np.ones((21, 21))), (0, 0), 8) * 1.6, 0, 1)[..., None]
G1 = gauss(A1, 340, 860, 330, 650)
G4 = gauss(A4, 720, 1350, 520, 900)
GB = gauss(A3, 990, 1150, 260)
yy3, xx3 = grid(A3); ANG3 = np.arctan2(yy3 - 1150, xx3 - 990)

def s1(t, closed):
    dy = -9 * math.sin(t * 1.6) * G1; dx = 4 * math.sin(t * 1.1) * G1
    img = warp(A1, dx, dy)
    if closed: img = img * (1 - EYE) + warp(A2, dx, dy) * EYE
    img = add_glow(img, 110, 240, 80, (1, 0.75, 0.4), 0.10 + 0.04 * math.sin(t * 13))
    img = add_glow(img, 760, 400, 260, (0.3, 0.55, 1.0), 0.05 + 0.02 * math.sin(t * 1.3))
    return img
def s3(t):
    w = 0.035 * math.sin(t * 7) * GB
    dx = (xx3 - 990) * w * np.cos(2 * ANG3 + t * 3); dy = (yy3 - 1150) * w * np.sin(2 * ANG3 + t * 3)
    img = warp(A3, dx, dy - 5 * math.sin(t * 1.5))
    return img
def s4(t):
    dy = -10 * math.sin(t * 1.7) * G4; dx = 5 * math.sin(t * 1.2) * G4
    return warp(A4, dx, dy)

def camera(img, cx, cy, zoom, jit):
    iw = img.shape[1]; s = (W / iw) * zoom
    M = cv2.getRotationMatrix2D((cx, cy), jit[2], s); M[0, 2] += W / 2 - cx + jit[0]; M[1, 2] += H / 2 - cy + jit[1]
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT), M
def scr(M, x, y): return M[0, 0] * x + M[0, 1] * y + M[0, 2], M[1, 0] * x + M[1, 1] * y + M[1, 2]

rng = np.random.default_rng(11)
DROPS = [(rng.uniform(-60, 60), rng.uniform(0, 1), rng.uniform(10, 22), rng.uniform(0, 6.3)) for _ in range(6)]
def water_drop(ctx, x, y, r, a=1.0):
    g = cairo.RadialGradient(x - r * 0.35, y - r * 0.35, r * 0.1, x, y, r)
    g.add_color_stop_rgba(0, 1, 1, 1, 0.85 * a); g.add_color_stop_rgba(0.5, 0.75, 0.88, 1, 0.35 * a); g.add_color_stop_rgba(1, 0.6, 0.75, 0.9, 0.55 * a)
    ctx.new_path(); ctx.arc(x, y, r, 0, 6.283); ctx.set_source(g); ctx.fill()
    ctx.new_path(); ctx.arc(x - r * 0.35, y - r * 0.35, r * 0.22, 0, 6.283); ctx.set_source_rgba(1, 1, 1, 0.9 * a); ctx.fill()

CUTS = [3.0, 10.0, 17.0, 24.0, END]
def frame(tr):
    t = math.floor(tr * STOP) / STOP
    r = random.Random(int(t * STOP))
    jit = (r.uniform(-1.1, 1.1), r.uniform(-1.1, 1.1), r.uniform(-0.05, 0.05))
    expo = 1 + r.uniform(-0.012, 0.012)
    M = None
    if t < 3.0:
        img = s3(t); z = 1.48 - 0.23 * ease(prog(t, 0, 0.6)) + 0.12 * ease(prog(t, 0.6, 3))
        out, M = camera(img, 900, 1150, z, jit)
    elif t < 10.0:
        closed = (5.4 <= t < 5.57) or (8.2 <= t < 9.6)
        img = s1(t, closed); p = ease(prog(t, 3, 10))
        out, M = camera(img, 470 - 40 * p, 836 - 120 * p, 1.0 + 0.3 * p, jit)
    elif t < 17.0:
        img = s3(t); p = ease(prog(t, 10, 17))
        out, M = camera(img, 719 + 120 * p, 1288 - 60 * p, 1.0 + 0.2 * p, jit)
    elif t < 24.0:
        img = s4(t); p = ease(prog(t, 17, 24))
        out, M = camera(img, 719 + 80 * p, 1288 - 150 * p, 1.0 + 0.22 * p, jit)
    elif t < END:
        img = s1(t, t > 25.4); p = ease(prog(t, 24, END))
        out, M = camera(img, 470, 836, 1.25 - 0.2 * p, jit)
    else:
        out, M = endcard.background(t - END)
    out = out * expo * VIG
    out = out + np.random.default_rng(int(t * STOP)).normal(0, 0.016, (H, W, 1)).astype(np.float32)
    s = surf_from(out); ctx = cairo.Context(s)
    k = M[0, 0]
    if 10.0 <= t < 17.0 or t < 3.0:
        x, y = scr(M, 1250 + 30 * math.sin(t * 1.3), 1330 - 20 * t % 60); water_drop(ctx, x, y, 12 * k / 0.75)
    if 17.0 <= t < 24.0:
        for (ox, ph, rr, w) in DROPS:
            q = (ph + (t - 17) * 0.18) % 1
            x, y = scr(M, 1000 + ox + 40 * math.sin(t * 1.4 + w) - q * 220, 1120 - q * 520)
            water_drop(ctx, x, y, rr * k / 0.75 * (0.7 + 0.6 * q), a=min(1, (1 - q) * 2))
    if t >= END:
        endcard.overlay(ctx, t - END, M, lambda lines, t0: label(ctx, lines, 540, 1585, t, END + t0, 14, size=60,
                        col=[(0.16, 0.11, 0.08), (0.2, 0.5, 0.1)]))
    L = label
    L(ctx, ["Uzayda ağlarsan", "gözyaşın nereye gider?"], 540, 1420, t, -0.3, 40, size=62, t1=1.6,
      col=[(0.16, 0.11, 0.08), (0.75, 0.12, 0.1)])
    L(ctx, ["Bu bir gözyaşı.", "Ve hiçbir yere gitmiyor."], 540, 1420, t, 1.6, 41, size=58, t1=3.0,
      col=[(0.16, 0.11, 0.08), (0.75, 0.12, 0.1)])
    L(ctx, ["Uzay istasyonu."], 300, 250, t, 3.2, 42, size=58, t1=10.0)
    L(ctx, ["Pencereden Dünya", "görünüyor..."], 540, 1480, t, 4.0, 43, size=54, t1=7.6)
    L(ctx, ["Duygulanmamak", "elde değil."], 540, 1480, t, 7.7, 44, size=56, t1=10.0)
    L(ctx, ["Dünya'da gözyaşını aşağı", "çeken şey yerçekimi."], 540, 260, t, 10.2, 45, size=50, t1=13.3)
    L(ctx, ["Uzayda gözyaşı düşmez.", "Gözünde su topu olur!"], 540, 260, t, 13.4, 46, size=54, t1=17.0,
      col=[(0.16, 0.11, 0.08), (0.12, 0.4, 0.75)])
    L(ctx, ["2013'te gerçek astronot Chris", "Hadfield bunu istasyonda denedi."], 540, 260, t, 17.2, 47, size=44, t1=20.4)
    L(ctx, ["Gözüne su sıktı, su gözünde", "top olup kaldı. Bizimki de denedi!"], 540, 260, t, 20.5, 48, size=44, t1=24.0)
    L(ctx, ["Yani uzayda ağlamak serbest..."], 540, 260, t, 24.2, 49, size=52, t1=END)
    L(ctx, ["...ama yanına peçete al."], 540, 1480, t, 25.4, 50, size=56, t1=END, col=(0.12, 0.4, 0.75))
    for c in CUTS:
        if c <= tr < c + 1 / 30: ctx.set_source_rgba(1, 1, 1, 0.35); ctx.paint()
    s.flush(); return s

def render(t0, t1, out):
    p = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgra', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                          '-c:v', 'libx264', '-crf', '20', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
    cache = {}
    for n in range(round(t0 * FPS), round(t1 * FPS)):
        tr = n / FPS; key = math.floor(tr * STOP)
        if key not in cache or any(c <= tr < c + 1 / 30 for c in CUTS):
            cache = {key: bytes(frame(tr).get_data())}
        p.stdin.write(cache[key])
    p.stdin.close(); return p.wait()

if __name__ == '__main__':
    if sys.argv[1] == 'still':
        for ts in sys.argv[2:]: frame(float(ts)).write_to_png(f'st_{ts}.png')
    else:
        sys.exit(render(float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]))
