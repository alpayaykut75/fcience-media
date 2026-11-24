"""Fcience – Mikrodalga fırın (Percy Spencer) kil videosu."""
import sys, math, random, subprocess, numpy as np, cv2, cairo
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _ROOT = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path.insert(0, _ROOT + "/lib"); _sys.path.insert(0, _ROOT + "/kapanis")
import endcard
from fcience import label, surf_from, F, back, prog, ease, clamp, glow_dot, VIG, W, H, FPS, STOP

DUR = 30.0
END = 27.2
def ld(i): return cv2.cvtColor(cv2.imread(f'{_HERE}/m{i}.png'), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
I = {i: ld(i) for i in range(1, 6)}
IH, IW = I[1].shape[:2]
yy, xx = np.mgrid[0:IH, 0:IW].astype(np.float32)
def gauss(cx, cy, sx, sy=None):
    sy = sy or sx; return np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2)
def rot_disp(cx, cy, ang, w):
    c, s = math.cos(ang), math.sin(ang)
    return ((xx - cx) * (c - 1) - (yy - cy) * s) * w, ((xx - cx) * s + (yy - cy) * (c - 1)) * w
def warp(img, dx, dy): return cv2.remap(img, xx - dx, yy - dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
def add_glow(img, cx, cy, r, col, amt): return img + gauss(cx, cy, r)[..., None] * np.array(col, np.float32) * amt

d = np.abs(I[1] - I[2]).mean(2); EYE = np.zeros_like(d); EYE[505:600, 315:465] = d[505:600, 315:465] > 0.08
EYE = np.clip(cv2.GaussianBlur(cv2.dilate(EYE, np.ones((17, 17))), (0, 0), 6) * 1.6, 0, 1)[..., None]

G = dict(h1=gauss(375, 520, 110, 140), b1=gauss(380, 900, 260, 420),
         h3=gauss(300, 470, 110, 130), b3=gauss(330, 900, 250, 420), bag=gauss(490, 700, 110, 130),
         e4=gauss(190, 850, 200, 420), c4=gauss(545, 680, 190, 260), egg=gauss(500, 990, 170, 150),
         chef=gauss(190, 820, 210, 480), hand5=gauss(370, 660, 90, 70))

def camera(img, cx, cy, zoom, jit):
    s = (W / IW) * zoom
    M = cv2.getRotationMatrix2D((cx, cy), jit[2], s); M[0, 2] += W / 2 - cx + jit[0]; M[1, 2] += H / 2 - cy + jit[1]
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT), M
def scr(M, x, y): return M[0, 0] * x + M[0, 1] * y + M[0, 2], M[1, 0] * x + M[1, 1] * y + M[1, 2]
def sc(M): return M[0, 0] / (W / IW) * (W / IW)

def lab_scene(t, k):
    img = I[k]
    if k == 1:
        dx, dy = rot_disp(380, 640, math.radians(1.5 * math.sin(t * 1.9)), G['h1'])
        dy = dy - 3 * math.sin(t * 2.5) * G['b1']
        img2 = warp(img, dx, dy)
        if any(b <= t < b + 0.17 for b in (5.0, 8.6)):
            img2 = img2 * (1 - EYE) + warp(I[2], dx, dy) * EYE
        img = img2
    elif k == 3:
        dx, dy = rot_disp(300, 600, math.radians(2.0 * math.sin(t * 3)), G['h3'])
        dy = dy - 5 * abs(math.sin(t * 4)) * G['b3'] - 6 * math.sin(t * 9) * G['bag']
        img = warp(img, dx, dy)
    elif k == 4:
        dy = -10 * math.sin(t * 6) * G['e4'] - 10 * math.sin(t * 6 + 1.3) * G['c4']
        img = warp(img, 0 * dy, dy)
    # radar + tubes flicker
    f = int(t * STOP)
    fl = 0.18 + 0.08 * random.Random(f).random()
    for (x, y) in ((735, 350), (805, 340), (890, 335), (725, 915), (775, 915), (825, 915)):
        img = add_glow(img, x, y, 30, (1.0, 0.6, 0.25), fl)
    img = add_glow(img, 805, 585, 120, (0.2, 1.0, 0.3), 0.10 + 0.05 * math.sin(t * 5))
    img = add_glow(img, 148, 300, 70, (1.0, 0.8, 0.45), 0.10 + 0.03 * math.sin(t * 17))
    return img

def kitchen(t):
    img = I[5]
    dx, dy = rot_disp(190, 1300, math.radians(1.6 * math.sin(t * 2.2)), G['chef'])
    dy = dy - 4 * math.sin(t * 5) * G['hand5']
    img = warp(img, dx, dy)
    img = add_glow(img, 680, 520, 90, (1.0, 0.6, 0.25), 0.18 + 0.08 * math.sin(t * 6))
    for i, x in enumerate((588, 682, 778)):
        on = (int(t * 3) + i) % 3 == 0
        img = add_glow(img, x, 760, 20, (1.0, 0.55, 0.15), 0.5 if on else 0.08)
    return img

rng = np.random.default_rng(5)
DRIPS = [(rng.uniform(-14, 14), rng.uniform(0, 1), rng.uniform(5, 9)) for _ in range(5)]
STEAM = [(rng.uniform(-30, 30), rng.uniform(0, 1), rng.uniform(14, 26)) for _ in range(7)]
CUTS = [3.0, 9.5, 15.5, 21.0, END]

def radar_sweep(ctx, M, t):
    cx, cy = scr(M, 805, 585); r = 88 * sc(M)
    a = (t * 2.4) % (2 * math.pi)
    for k in range(10):
        aa = a - k * 0.06
        ctx.new_path(); ctx.move_to(cx, cy); ctx.line_to(cx + r * math.cos(aa), cy + r * math.sin(aa))
        ctx.set_source_rgba(0.55, 1, 0.45, 0.5 * (1 - k / 10)); ctx.set_line_width(4 * sc(M)); ctx.stroke()
    blip = (t * 0.7) % 1
    bx, by = cx + r * 0.5 * math.cos(1.0), cy + r * 0.5 * math.sin(1.0)
    glow_dot(ctx, bx, by, 5 * sc(M), (0.6, 1, 0.5), 0.9 * (1 - blip))

def drips(ctx, M, t, x0=428, y0=950):
    for (ox, ph, rr) in DRIPS:
        k = (ph + t * 0.6) % 1
        x, y = scr(M, x0 + ox, y0 + k * 160)
        ctx.new_path(); ctx.save(); ctx.translate(x, y); ctx.scale(rr * sc(M), rr * 1.4 * sc(M)); ctx.arc(0, 0, 1, 0, 6.283); ctx.restore()
        ctx.set_source_rgba(0.36, 0.2, 0.1, 0.9 * (1 - k)); ctx.fill()

def steam(ctx, M, t):
    for (ox, ph, rr) in STEAM:
        k = (ph + t * 0.35) % 1
        x, y = scr(M, 370 + ox + 18 * math.sin(t * 2 + ph * 9), 630 - k * 200)
        ctx.new_path(); ctx.arc(x, y, rr * (0.6 + k) * sc(M), 0, 6.283)
        ctx.set_source_rgba(1, 1, 1, 0.22 * math.sin(math.pi * k)); ctx.fill()

def frame(tr):
    t = math.floor(tr * STOP) / STOP
    r = random.Random(int(t * STOP))
    jit = (r.uniform(-1.1, 1.1), r.uniform(-1.1, 1.1), r.uniform(-0.05, 0.05))
    expo = 1 + r.uniform(-0.012, 0.012)
    if t < 3.0:
        img = lab_scene(t, 1); z = 2.1 + 0.15 * ease(prog(t, 0, 3))
        out, M = camera(img, 430, 900, z, jit)
    elif t < 9.5:
        img = lab_scene(t, 1); p = ease(prog(t, 3.0, 9.5))
        out, M = camera(img, 470 - 20 * p, 836 + 20 * p, 1.0 + 0.18 * p, jit)
    elif t < 15.5:
        img = lab_scene(t, 3); p = ease(prog(t, 9.5, 15.5))
        shake = 6 * math.sin(t * 50) * max(0, 1 - (t - 9.9) * 3) if t > 9.9 else 0
        out, M = camera(img, 470, 836 - 60 * p, 1.02 + 0.12 * p, (jit[0] + shake, jit[1], jit[2]))
    elif t < 21.0:
        img = lab_scene(t, 4); p = ease(prog(t, 15.5, 21.0))
        hit = max(0, 1 - (t - 16.8) * 2.5) if t > 16.8 else 0
        out, M = camera(img, 470, 880, 1.0 + 0.1 * p + 0.05 * hit, (jit[0] + 12 * hit * math.sin(t * 60), jit[1], jit[2]))
        if 16.8 <= t < 16.95: out = out * 0.6 + 0.4
    elif t < END:
        img = kitchen(t); p = ease(prog(t, 21.0, END))
        out, M = camera(img, 470, 836 - 40 * p, 1.0 + 0.12 * p, jit)
    else:
        out, M = endcard.background(t - END)
    out = out * expo * VIG
    out = out + np.random.default_rng(int(t * STOP)).normal(0, 0.016, (H, W, 1)).astype(np.float32)
    s = surf_from(out); ctx = cairo.Context(s)
    if t < 21.0: radar_sweep(ctx, M, t)
    if t < 9.5: drips(ctx, M, t)
    if 21.0 <= t < END: steam(ctx, M, t)
    if t >= END:
        endcard.overlay(ctx, t - END, M, lambda lines, t0: label(ctx, lines, 540, 1585, t, END + t0, 14, size=60,
                        col=[(0.16, 0.11, 0.08), (0.2, 0.5, 0.1)]))
    # ---- text ----
    L = label
    L(ctx, ["Cebindeki çikolata eridi...", "ve mutfağın yıldızı doğdu."], 540, 1380, t, 0.15, 21, size=54, t1=3.0,
      col=[(0.16, 0.11, 0.08), (0.75, 0.12, 0.1)])
    L(ctx, ["ABD, 1945."], 280, 250, t, 3.2, 22, size=60, t1=9.5)
    L(ctx, ["Mühendis Percy Spencer", "radar makinesinin başında."], 540, 420, t, 4.1, 23, size=50, t1=6.6)
    L(ctx, ["Bir anda cebindeki", "çikolatanın eridiğini görüyor."], 540, 420, t, 6.7, 24, size=50, t1=9.5)
    L(ctx, ["Merak ediyor: makinenin", "yanına mısır koyuyor."], 540, 280, t, 9.7, 25, size=50, t1=12.4)
    L(ctx, ["Patlamış mısır", "her yere saçılıyor!"], 540, 280, t, 12.5, 26, size=58, t1=15.5)
    L(ctx, ["Sonra bir yumurta", "deniyorlar..."], 540, 280, t, 15.6, 27, size=56, t1=16.8)
    L(ctx, ["PAT!"], 800, 1260, t, 16.8, 28, size=110, t1=18.2, col=(0.75, 0.12, 0.1))
    L(ctx, ["Yumurta patlıyor,", "bilim ilerliyor."], 540, 280, t, 18.3, 29, size=56, t1=21.0)
    L(ctx, ["İlk mikrodalga fırın:", "buzdolabı boyunda, 300 kilodan ağır."], 540, 250, t, 21.2, 30, size=44, t1=24.2)
    L(ctx, ["Bugün 2 dakikada ısıtıyoruz.", "Teşekkürler, erimiş çikolata!"], 540, 250, t, 24.3, 31, size=48, t1=END,
      col=[(0.16, 0.11, 0.08), (0.2, 0.5, 0.1)])
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
