"""Fcience – Marie Curie claymation post (cut-out/warp animation of still images)."""
import sys, math, random, subprocess, numpy as np, cv2, cairo
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _ROOT = _os.path.abspath(_os.path.join(_HERE, "..", ".."))
_sys.path.insert(0, _ROOT + "/lib"); _sys.path.insert(0, _ROOT + "/kapanis")
import endcard
_os.chdir(_HERE)  # görseller ve ses bu klasörde

W, H, FPS, STOP = 1080, 1920, 30, 12
DUR = 30.0
IMG = {k: cv2.cvtColor(cv2.imread(f'{k}.jpg'), cv2.COLOR_BGR2RGB).astype(np.float32) / 255 for k in 'ABCD'}
IH, IW = IMG['A'].shape[:2]  # 2576 x 1438

# ---------- helpers ----------
def clamp(x, a=0., b=1.): return max(a, min(b, x))
def prog(t, a, b): return clamp((t - a) / (b - a))
def ease(x): return x * x * (3 - 2 * x)
def back(x, s=1.7): x -= 1; return x * x * ((s + 1) * x + s) + 1
yy, xx = np.mgrid[0:IH, 0:IW].astype(np.float32)

def gauss(cx, cy, sx, sy=None):
    sy = sy or sx
    return np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2)

# precomputed weight maps
G_HEAD = gauss(735, 1000, 230, 260)
G_TORSO = gauss(720, 1650, 450, 420)
G_FLASK = gauss(1120, 1230, 170, 230)
G_VIS = gauss(1250, 1500, 260, 650)
G_CLIP = gauss(1000, 1700, 200, 200)
G_BOX = gauss(500, 1700, 330, 260)

def rot_disp(cx, cy, ang, w):
    c, s = math.cos(ang), math.sin(ang)
    dx = (xx - cx) * (c - 1) - (yy - cy) * s
    dy = (xx - cx) * s + (yy - cy) * (c - 1)
    return dx * w, dy * w

def warp(img, dx, dy):
    return cv2.remap(img, xx - dx, yy - dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

# eye-blink mask (A vs D differ only around eyes)
diff = np.abs(IMG['A'] - IMG['D']).mean(2)
EYE = np.zeros_like(diff); EYE[900:1120, 560:920] = (diff[900:1120, 560:920] > 0.08)
EYE = cv2.GaussianBlur(cv2.dilate(EYE, np.ones((25, 25))), (0, 0), 9)[..., None]
EYE = np.clip(EYE * 1.6, 0, 1)

# gauge needle (C): inpaint original needle, redraw later
PIV = (1117, 1172); TIP0 = (1072, 1120)
Cu8 = (IMG['C'] * 255).astype(np.uint8)
nm = np.zeros((IH, IW), np.uint8); cv2.line(nm, PIV, TIP0, 255, 14)
cv2.circle(nm, PIV, 6, 0, -1)
C_CLEAN = cv2.inpaint(Cu8, nm, 7, cv2.INPAINT_TELEA).astype(np.float32) / 255
NEEDLE_LEN = math.hypot(TIP0[0] - PIV[0], TIP0[1] - PIV[1]) + 6

rng = np.random.default_rng(7)
PART = [(rng.uniform(380, 1000), rng.uniform(1350, 1900), rng.uniform(2, 6), rng.uniform(0, 6.28), rng.uniform(40, 110)) for _ in range(90)]
BUB = [(rng.uniform(-60, 60), rng.uniform(0, 1), rng.uniform(5, 11), rng.uniform(0.6, 1.4)) for _ in range(18)]
DUST = [(rng.uniform(0, IW), rng.uniform(0, IH), rng.uniform(1.5, 3.5), rng.uniform(0, 6.28)) for _ in range(60)]

def add_glow(img, cx, cy, r, col, amt):
    g = gauss(cx, cy, r)[..., None] if r > 0 else 0
    return img + g * np.array(col, np.float32) * amt

# ---------- camera ----------
def camera(img, cx, cy, zoom, jitter):
    s = (W / IW) * zoom
    jx, jy, jr = jitter
    M = cv2.getRotationMatrix2D((cx, cy), jr, s)
    M[0, 2] += W / 2 - cx + jx; M[1, 2] += H / 2 - cy + jy
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT), M

# ---------- per-scene base images ----------
def scene_A(t):
    img = IMG['A']
    dx, dy = rot_disp(735, 1230, math.radians(1.6 * math.sin(t * 1.7)), G_HEAD)
    dy = dy - 4 * math.sin(t * 2.4) * G_TORSO
    fx, fy = rot_disp(1200, 1520, math.radians(2.2 * math.sin(t * 2.1 + 1)), G_FLASK)
    img = warp(img, dx + fx, dy + fy)
    blink = 0.0
    for b in (5.1, 8.4):
        if b <= t < b + 0.17: blink = 1.0
    if 9.5 <= t < 10.6: blink = 1.0
    if blink:
        imgD = warp(IMG['D'], dx + fx, dy + fy)
        img = img * (1 - EYE) + imgD * EYE
    pulse = 0.55 + 0.25 * math.sin(t * 4.2) + 0.35 * prog(t, 7.2, 7.8)
    img = add_glow(img, 1115, 1290, 190, (0.35, 1.0, 0.25), 0.35 * pulse)
    img = add_glow(img, 160, 1000, 120, (1.0, 0.8, 0.45), 0.12 * (1 + 0.15 * math.sin(t * 23)))
    return img, dict(flask=(1115 + 0, 1290), pulse=pulse)

def scene_B(t, close=False):
    img = IMG['B']
    sh = math.sin(t * 38) * (3.5 if t > 23.2 or close else 1.8)
    dx = sh * G_VIS
    dy = -3 * math.sin(t * 2.2) * G_BOX + 2.5 * math.sin(t * 31) * G_CLIP
    img = warp(img, dx, dy)
    fl = 1.0 if (int(t * 12) % 23) else 0.93
    img = img * fl
    return img, {}

def scene_C(t):
    img = C_CLEAN.copy()
    pulse = 0.6 + 0.25 * math.sin(t * 3.7) + 0.4 * prog(t, 13.0, 13.8)
    img = add_glow(img, 620, 1650, 330, (0.4, 1.0, 0.35), 0.22 * pulse)
    img = add_glow(img, 870, 880, 220, (0.35, 1.0, 0.3), 0.15 * pulse)
    return img, dict(pulse=pulse)

def needle_angle(t):
    base = math.atan2(TIP0[1] - PIV[1], TIP0[0] - PIV[0])
    lvl = 0.15 + 0.7 * ease(prog(t, 12.4, 13.8))
    jit = 0.06 * math.sin(t * 37) + 0.04 * math.sin(t * 59)
    a_min, a_max = math.radians(-150), math.radians(-32)
    return a_min + (a_max - a_min) * clamp(lvl + jit)

# ---------- cairo overlay helpers ----------
def surf_from(arr):
    a = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    buf = np.ndarray((H, W, 4), np.uint8, s.get_data())
    buf[..., 0] = a[..., 2]; buf[..., 1] = a[..., 1]; buf[..., 2] = a[..., 0]; buf[..., 3] = 255
    s.mark_dirty(); return s

def F(ctx, size, fam='Poppins', bold=True):
    ctx.select_font_face(fam, 0, 1 if bold else 0); ctx.set_font_size(size)

def label(ctx, lines, cx, cy, t, t0, seed, size=56, col=(0.16, 0.11, 0.08), paper=(0.96, 0.92, 0.82), t1=None):
    if t < t0 or (t1 is not None and t >= t1): return
    r = random.Random(seed + int(t * STOP) * 7)
    a = back(prog(t, t0, t0 + 0.2)) if t < t0 + 0.2 else 1.0
    if t1 is not None and t > t1 - 0.12: a = max(0.0, (t1 - t) / 0.12)
    F(ctx, size)
    wds = [ctx.text_extents(l).x_advance for l in lines]
    w = max(wds) + 70; h = size * 1.32 * len(lines) + 46
    rot = random.Random(seed).uniform(-0.035, 0.035) + r.uniform(-0.004, 0.004)
    ctx.save(); ctx.translate(cx + r.uniform(-0.7, 0.7), cy + r.uniform(-0.7, 0.7)); ctx.rotate(rot); ctx.scale(a, a)
    # torn paper polygon
    pr = random.Random(seed)
    pts = []
    n = 14
    for i in range(n + 1): pts.append((-w / 2 + w * i / n, -h / 2 + pr.uniform(-4, 4)))
    for i in range(4): pts.append((w / 2 + pr.uniform(-4, 3), -h / 2 + h * (i + 1) / 4))
    for i in range(n + 1): pts.append((w / 2 - w * i / n, h / 2 + pr.uniform(-4, 4)))
    for i in range(4): pts.append((-w / 2 + pr.uniform(-3, 4), h / 2 - h * (i + 1) / 4))
    ctx.save(); ctx.translate(6, 9)
    ctx.move_to(*pts[0]); [ctx.line_to(*p) for p in pts[1:]]; ctx.close_path(); ctx.set_source_rgba(0, 0, 0, 0.35); ctx.fill(); ctx.restore()
    ctx.move_to(*pts[0]); [ctx.line_to(*p) for p in pts[1:]]; ctx.close_path(); ctx.set_source_rgba(*paper, 1); ctx.fill()
    for i, l in enumerate(lines):
        F(ctx, size); e = ctx.text_extents(l)
        ctx.move_to(-e.x_advance / 2, -h / 2 + 23 + size * 1.05 + i * size * 1.32)
        c = col if not (isinstance(col, list)) else col[i]
        ctx.set_source_rgba(*c, 1); ctx.show_text(l); ctx.new_path()
    ctx.restore()

def bubble(ctx, text_lines, x, y, tail, t, t0, t1, size=46):
    if not (t0 <= t < t1): return
    a = back(prog(t, t0, t0 + 0.2)) if t < t0 + 0.2 else 1.0
    F(ctx, size)
    w = max(ctx.text_extents(l).x_advance for l in text_lines) + 70; h = size * 1.3 * len(text_lines) + 44
    ctx.save(); ctx.translate(x, y); ctx.scale(a, a)
    def path():
        ctx.new_path(); r = 40
        ctx.arc(w / 2 - r, -h / 2 + r, r, -math.pi / 2, 0); ctx.arc(w / 2 - r, h / 2 - r, r, 0, math.pi / 2)
        ctx.arc(-w / 2 + r, h / 2 - r, r, math.pi / 2, math.pi); ctx.arc(-w / 2 + r, -h / 2 + r, r, math.pi, 1.5 * math.pi); ctx.close_path()
        tx, ty = tail; ctx.move_to(tx - 26, h / 2 - 4); ctx.line_to(tx + 40, ty); ctx.line_to(tx + 26, h / 2 - 4); ctx.close_path()
    path(); ctx.set_source_rgba(0.12, 0.09, 0.07, 1); ctx.set_line_width(10); ctx.set_line_join(1); ctx.stroke()
    path(); ctx.set_source_rgba(1, 0.99, 0.96, 1); ctx.fill()
    for i, l in enumerate(text_lines):
        e = ctx.text_extents(l); ctx.move_to(-e.x_advance / 2, -h / 2 + 22 + size + i * size * 1.3)
        ctx.set_source_rgba(0.12, 0.09, 0.07, 1); ctx.show_text(l); ctx.new_path()
    ctx.restore()

def glow_dot(ctx, x, y, r, col, a):
    g = cairo.RadialGradient(x, y, 0, x, y, r * 3)
    g.add_color_stop_rgba(0, *col, a); g.add_color_stop_rgba(0.35, *col, a * 0.5); g.add_color_stop_rgba(1, *col, 0)
    ctx.set_source(g); ctx.arc(x, y, r * 3, 0, 6.283); ctx.fill()

def to_screen(M, x, y):
    return M[0, 0] * x + M[0, 1] * y + M[0, 2], M[1, 0] * x + M[1, 1] * y + M[1, 2]

# ---------- grain / vignette ----------
VIG = (1 - 0.35 * ((np.mgrid[0:H, 0:W][1] / W - 0.5) ** 2 * 2.2 + (np.mgrid[0:H, 0:W][0] / H - 0.5) ** 2 * 1.6))[..., None].astype(np.float32)


# ---------- frame ----------
CUTS = [3.2, 11.0, 17.5, 25.0, 27.2]
def frame(tr):
    t = math.floor(tr * STOP) / STOP  # stop-motion: 12 unique poses / sec
    r = random.Random(int(t * STOP))
    jit = (r.uniform(-1.1, 1.1), r.uniform(-1.1, 1.1), r.uniform(-0.05, 0.05))
    expo = 1 + r.uniform(-0.012, 0.012)
    if t < 3.2:
        img, info = scene_B(t, close=True)
        z = 1.75 + 0.12 * ease(prog(t, 0, 3.2))
        out, M = camera(img, 680, 1720, z, jit)
    elif t < 11.0:
        img, info = scene_A(t)
        p = ease(prog(t, 3.2, 11.0))
        out, M = camera(img, 719 + 140 * p, 1288 - 110 * p, 1.0 + 0.32 * p, jit)
    elif t < 17.5:
        img, info = scene_C(t)
        p = ease(prog(t, 11.0, 17.5))
        out, M = camera(img, 719 + 60 * p, 1288 + 60 * p, 1.02 + 0.22 * p, jit)
    elif t < 25.0:
        img, info = scene_B(t)
        p = ease(prog(t, 17.5, 25.0))
        out, M = camera(img, 719, 1288 + 80 * p, 1.0 + 0.1 * p, jit)
    elif t < 27.2:
        img = cv2.GaussianBlur(C_CLEAN, (0, 0), 18) * 0.32
        out, M = camera(img, 719, 1288, 1.0, (0, 0, 0)); info = {}
    else:
        out, M = endcard.background(t - 27.2); info = {}
    out = out * expo * VIG
    out = out + np.random.default_rng(int(t * STOP)).normal(0, 0.018, (H, W, 1)).astype(np.float32)
    s = surf_from(out); ctx = cairo.Context(s)

    # particles / overlays per scene
    if 11.0 <= t < 17.5 or t < 3.2:
        pass
    if 11.0 <= t < 17.5:
        ang = needle_angle(t)
        px, py = to_screen(M, *PIV)
        tx, ty = to_screen(M, PIV[0] + NEEDLE_LEN * math.cos(ang), PIV[1] + NEEDLE_LEN * math.sin(ang))
        ctx.move_to(px, py); ctx.line_to(tx, ty); ctx.set_source_rgba(0.12, 0.08, 0.06, 1); ctx.set_line_width(5.5 * M[0, 0] / (W / IW)); ctx.set_line_cap(1); ctx.stroke()
        ctx.arc(px, py, 9 * M[0, 0] / (W / IW), 0, 6.283); ctx.set_source_rgba(0.15, 0.1, 0.07, 1); ctx.fill()
        for (x0, y0, rr, ph, sp) in PART:
            yv = y0 - ((t - 11.0) * sp) % 700
            xv = x0 + 25 * math.sin(t * 1.5 + ph)
            a = 0.8 * (0.5 + 0.5 * math.sin(t * 5 + ph)) * clamp((yv - 1150) / 300)
            sx, sy = to_screen(M, xv, yv)
            glow_dot(ctx, sx, sy, rr * 0.9, (0.55, 1, 0.45), a)
    if 3.2 <= t < 11.0:
        pulse = info['pulse']
        for (ox, ph, rr, sp) in BUB:
            k = (ph + (t - 3.2) * 0.35 * sp) % 1
            bx, by = 1112 + ox * (1 - k * 0.5), 1320 - k * 110
            sx, sy = to_screen(M, bx, by)
            ctx.arc(sx, sy, rr * M[0, 0] / (W / IW) * 0.8, 0, 6.283); ctx.set_source_rgba(0.75, 1, 0.4, 0.55 * (1 - k)); ctx.fill()
        for (x0, y0, rr, ph) in DUST[:30]:
            sx, sy = to_screen(M, (x0 + t * 12) % IW, (y0 - t * 8) % IH)
            glow_dot(ctx, sx, sy, rr * 0.8, (1, 0.9, 0.7), 0.25 + 0.2 * math.sin(t * 2 + ph))

    if t >= 27.2:
        endcard.overlay(ctx, t - 27.2, M, lambda lines, t0: label(ctx, lines, 540, 1585, t, 27.2 + t0, 14, size=60,
                        col=[(0.16, 0.11, 0.08), (0.2, 0.5, 0.1)]))
    # ---- text ----
    label(ctx, ["Bu defteri okumak için", "önce imza atman gerekiyor."], 540, 330, t, 0.15, 1, size=58, t1=3.2,
          col=[(0.16, 0.11, 0.08), (0.75, 0.12, 0.1)])
    label(ctx, ["Paris, 1898."], 300, 250, t, 3.4, 2, size=60, t1=11.0)
    label(ctx, ["Marie ve Pierre Curie", "radyumu keşfediyor."], 540, 420, t, 4.4, 3, size=54, t1=7.2)
    label(ctx, ["Radyum karanlıkta", "yeşil yeşil parlıyor..."], 540, 420, t, 7.3, 4, size=54, t1=9.2)
    label(ctx, ["...ve Marie", "buna bayılıyor."], 540, 420, t, 9.25, 5, size=58, t1=11.0)
    label(ctx, ["Ama radyum", "her şeye bulaşıyor."], 540, 330, t, 11.2, 6, size=58, t1=13.5)
    label(ctx, ["Defterlerine bile."], 540, 330, t, 13.6, 7, size=64, t1=15.6, col=(0.7, 0.1, 0.08))
    label(ctx, ["Aradan 120 yıldan", "fazla geçiyor..."], 540, 330, t, 15.7, 8, size=58, t1=17.5)
    label(ctx, ["Bugün defterler Paris'te,", "kurşun kaplı kutularda."], 540, 250, t, 17.7, 9, size=50, t1=21.4)
    bubble(ctx, ["Önce şu formu", "imzalayın lütfen."], 430, 600, (-30, 120), t, 19.4, 21.6)
    label(ctx, ["Okumak isteyen feragatname", "imzalıyor, koruyucu kıyafet giyiyor."], 540, 250, t, 21.5, 10, size=40, t1=25.0)
    bubble(ctx, ["Ben sadece", "okuyacaktım ama?"], 700, 640, (60, 120), t, 23.0, 25.0)
    if 25.0 <= t < 27.2:
        label(ctx, ["Radyum-226'nın yarı ömrü:"], 540, 620, t, 25.15, 11, size=48)
        label(ctx, ["~1.600 yıl"], 540, 780, t, 25.6, 12, size=96, col=(0.12, 0.55, 0.15))
        label(ctx, ["Yani... acele etmeyin."], 540, 960, t, 26.5, 13, size=52)
        if False:
            a = back(prog(t, 27.4, 27.7))
            ctx.save(); ctx.translate(540, 1240); ctx.scale(a, a)
            F(ctx, 110, 'Archivo Black'); e = ctx.text_extents("FCIENCE")
            ctx.move_to(-e.x_advance / 2, 0); ctx.set_source_rgba(1, 1, 1, 1); ctx.show_text("FCIENCE"); ctx.new_path()
            F(ctx, 40); e = ctx.text_extents("Takip et, tuhaf bilimi kaçırma")
            ctx.move_to(-e.x_advance / 2, 75); ctx.set_source_rgba(0.6, 1, 0.5, 1); ctx.show_text("Takip et, tuhaf bilimi kaçırma"); ctx.new_path()
            ctx.restore()
    # cut flash (white frame) for snap
    for c in CUTS:
        if c <= tr < c + 1 / 30:
            ctx.set_source_rgba(1, 1, 1, 0.35); ctx.paint()
    s.flush()
    return s

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
        for ts in sys.argv[2:]:
            frame(float(ts)).write_to_png(f'st_{ts}.png')
    else:
        sys.exit(render(float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]))
