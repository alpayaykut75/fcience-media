"""Hikaye klibi: videonun ilk 12 sn'si + yırtık kâğıt "Tamamı profilde!" etiketi.
Kullanım: python3 tools/story.py VIDEO.mp4 CIKTI.mp4 ETIKET_Y BASLAMA_SN
 ETIKET_Y: etiketin dikey merkezi (videodaki altyazılarla çakışmasın! Marie 1440, gözyaşı 1690, mikrodalga 1440)
 BASLAMA_SN: etiketin belirdiği an (ör. 0.8)"""
import sys, cairo, random, subprocess, tempfile, os
W, H = 1080, 1920
def label_png(out, y):
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H); ctx = cairo.Context(s)
    lines = ["Tamamı profilde!", "Yeni bölüm yayında"]
    ctx.select_font_face('Poppins', 0, 1); ctx.set_font_size(64)
    w = max(ctx.text_extents(l).x_advance for l in lines) + 80; h = 64 * 1.32 * 2 + 50
    pr = random.Random(3); n = 14; pts = []
    for i in range(n + 1): pts.append((-w / 2 + w * i / n, -h / 2 + pr.uniform(-4, 4)))
    for i in range(4): pts.append((w / 2 + pr.uniform(-4, 3), -h / 2 + h * (i + 1) / 4))
    for i in range(n + 1): pts.append((w / 2 - w * i / n, h / 2 + pr.uniform(-4, 4)))
    for i in range(4): pts.append((-w / 2 + pr.uniform(-3, 4), h / 2 - h * (i + 1) / 4))
    ctx.translate(540, y); ctx.rotate(-0.02)
    ctx.save(); ctx.translate(6, 9); ctx.move_to(*pts[0]); [ctx.line_to(*p) for p in pts[1:]]; ctx.close_path(); ctx.set_source_rgba(0, 0, 0, 0.35); ctx.fill(); ctx.restore()
    ctx.move_to(*pts[0]); [ctx.line_to(*p) for p in pts[1:]]; ctx.close_path(); ctx.set_source_rgba(0.96, 0.92, 0.82, 1); ctx.fill()
    cols = [(0.2, 0.5, 0.1), (0.16, 0.11, 0.08)]
    for i, l in enumerate(lines):
        ctx.set_font_size(64 if i == 0 else 50); e = ctx.text_extents(l)
        ctx.move_to(-e.x_advance / 2, -h / 2 + 25 + 64 * 1.02 + i * 64 * 1.32); ctx.set_source_rgba(*cols[i], 1); ctx.show_text(l); ctx.new_path()
    s.write_to_png(out)
if __name__ == '__main__':
    vid, out, y, st = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
    lab = os.path.join(tempfile.mkdtemp(), 'lab.png'); label_png(lab, y)
    # -loop 1 -t 12 ŞART: tek kareli PNG geç fade-in olunca görünmez kalıyor
    fc = (f"[0:v]trim=0:12,setpts=PTS-STARTPTS[v];[1:v]format=rgba,fade=in:st={st}:d=0.3:alpha=1[l];"
          "[v][l]overlay=0:0:shortest=1,fade=out:st=11.4:d=0.6[o];[0:a]atrim=0:12,asetpts=PTS-STARTPTS,afade=out:st=11.2:d=0.8[a]")
    sys.exit(subprocess.call(['ffmpeg', '-y', '-v', 'error', '-i', vid, '-loop', '1', '-t', '12', '-framerate', '30', '-i', lab,
                              '-filter_complex', fc, '-map', '[o]', '-map', '[a]', '-c:v', 'libx264', '-crf', '22', '-preset', 'veryfast',
                              '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '160k', '-movflags', '+faststart', out]))
