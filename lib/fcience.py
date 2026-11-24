"""Fcience ortak animasyon kütüphanesi (kil / stop-motion kesme animasyonu).
Bölüm betikleri: sys.path'e bu klasörü ekleyip `from fcience import *` yapar.
"""
import math, random, subprocess, numpy as np, cv2, cairo
W, H, FPS, STOP = 1080, 1920, 30, 12   # çıktı 1080x1920, 30 fps; benzersiz kare 12 fps (stop-motion hissi)

def load(path):
    """Görseli float32 RGB 0..1 olarak yükle."""
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255

def clamp(x, a=0., b=1.): return max(a, min(b, x))
def prog(t, a, b): return clamp((t - a) / (b - a))
def ease(x): return x * x * (3 - 2 * x)
def back(x, s=1.7): x -= 1; return x * x * ((s + 1) * x + s) + 1

_GRIDS = {}
def grid(img):
    """Görsel boyutuna göre (yy, xx) ızgarası (önbellekli)."""
    k = img.shape[:2]
    if k not in _GRIDS: _GRIDS[k] = np.mgrid[0:k[0], 0:k[1]].astype(np.float32)
    return _GRIDS[k]

def gauss(img, cx, cy, sx, sy=None):
    """Görsel üzerinde gauss ağırlık haritası (warp bölgeleri için)."""
    yy, xx = grid(img); sy = sy or sx
    return np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2)

def rot_disp(img, cx, cy, ang, w):
    """(cx,cy) etrafında ang radyan döndürme yer değiştirmesi, w ağırlığıyla."""
    yy, xx = grid(img); c, s = math.cos(ang), math.sin(ang)
    dx = (xx - cx) * (c - 1) - (yy - cy) * s
    dy = (xx - cx) * s + (yy - cy) * (c - 1)
    return dx * w, dy * w

def warp(img, dx, dy):
    yy, xx = grid(img)
    return cv2.remap(img, xx - dx, yy - dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def add_glow(img, cx, cy, r, col, amt):
    return img + gauss(img, cx, cy, r)[..., None] * np.array(col, np.float32) * amt

def blink_mask(open_img, closed_img, box, thr=0.08):
    """Göz açık / kapalı varyantların farkından göz maskesi. box=(y0,y1,x0,x1)."""
    y0, y1, x0, x1 = box
    d = np.abs(open_img - closed_img).mean(2); m = np.zeros_like(d)
    m[y0:y1, x0:x1] = d[y0:y1, x0:x1] > thr
    m = cv2.GaussianBlur(cv2.dilate(m, np.ones((25, 25))), (0, 0), 9)[..., None]
    return np.clip(m * 1.6, 0, 1)

def jitter(seed):
    """Stop-motion kare titremesi (azaltılmış değerler — kullanıcı onaylı)."""
    r = random.Random(seed)
    return (r.uniform(-1.1, 1.1), r.uniform(-1.1, 1.1), r.uniform(-0.05, 0.05))

def camera(img, cx, cy, zoom, jit=(0, 0, 0)):
    """Görseli 1080x1920 kadraja yerleştir; (cx,cy) görsel koordinatında merkez. (out, M) döner."""
    IW = img.shape[1]; s = (W / IW) * zoom
    jx, jy, jr = jit
    M = cv2.getRotationMatrix2D((cx, cy), jr, s)
    M[0, 2] += W / 2 - cx + jx; M[1, 2] += H / 2 - cy + jy
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT), M

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

def water_drop(ctx, x, y, r, a=1.0):
    g = cairo.RadialGradient(x - r * 0.35, y - r * 0.35, r * 0.1, x, y, r)
    g.add_color_stop_rgba(0, 1, 1, 1, 0.85 * a); g.add_color_stop_rgba(0.5, 0.75, 0.88, 1, 0.35 * a); g.add_color_stop_rgba(1, 0.6, 0.75, 0.9, 0.55 * a)
    ctx.new_path(); ctx.arc(x, y, r, 0, 6.283); ctx.set_source(g); ctx.fill()
    ctx.new_path(); ctx.arc(x - r * 0.35, y - r * 0.35, r * 0.22, 0, 6.283); ctx.set_source_rgba(1, 1, 1, 0.9 * a); ctx.fill()

VIG = (1 - 0.35 * ((np.mgrid[0:H, 0:W][1] / W - 0.5) ** 2 * 2.2 + (np.mgrid[0:H, 0:W][0] / H - 0.5) ** 2 * 1.6))[..., None].astype(np.float32)

def finish(img, seed):
    """Pozlama titremesi + grain + vinyet (stop-motion film hissi)."""
    r = random.Random(seed)
    img = img * (1 + r.uniform(-0.012, 0.012))
    img = img + np.random.default_rng(seed).normal(0, 0.017, img.shape[:2])[..., None].astype(np.float32)
    return img * VIG

def cut_flash(ctx, tr, cuts):
    for c in cuts:
        if c <= tr < c + 1 / FPS:
            ctx.set_source_rgba(1, 1, 1, 0.35); ctx.paint()

def render(frame, cuts, t0, t1, out):
    """frame(tr)->cairo surface; 12 fps benzersiz kare, 30 fps çıktı; ffmpeg'e raw BGRA."""
    p = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgra', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                          '-c:v', 'libx264', '-crf', '20', '-preset', 'veryfast', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
    cache = {}
    for n in range(round(t0 * FPS), round(t1 * FPS)):
        tr = n / FPS; key = math.floor(tr * STOP)
        if key not in cache or any(c <= tr < c + 1 / FPS for c in cuts):
            cache = {key: bytes(frame(tr).get_data())}
        p.stdin.write(cache[key])
    p.stdin.close(); return p.wait()
