import numpy as np, wave
SR=44100; D=30.0; N=int(SR*D); mix=np.zeros(N); rng=np.random.default_rng(12)
def tt(d): return np.arange(int(SR*d))/SR
def add(s,at,g=1.):
    i=int(at*SR)
    if i>=N: return
    j=min(N,i+len(s)); mix[i:j]+=s[:j-i]*g
def nf(m): return 440*2**((m-69)/12)
def bell(f,d=1.6):
    t=tt(d); return (np.sin(2*np.pi*f*t)+0.3*np.sin(2*np.pi*f*2.01*t))*np.exp(-t*2.5)
# dreamy space pad
t=tt(D)
for f in (nf(57),nf(64),nf(69),nf(72)):
    mix+=0.018*np.sin(2*np.pi*f*t+np.sin(2*np.pi*0.2*t))*(0.6+0.4*np.sin(2*np.pi*0.07*t+f))
# music box melody (soft, then playful after 10s)
mel=[76,79,83,81,79,76,74,76, 72,74,76,79,81,79,76,74]
beat=0.42; k=0
while k*beat<D-0.5:
    at=k*beat; m=mel[k%len(mel)]
    g=0.06 if at<10 or at>24 else 0.075
    if any(c-0.2<at<c+0.1 for c in (3.0,10.0,17.0,24.0)): g*=0.4
    add(bell(nf(m)),at,g); k+=1
def plink(f=1800,d=0.25):
    t=tt(d); ff=f*(1+2*np.exp(-t*40)); return np.sin(2*np.pi*np.cumsum(ff)/SR)*np.exp(-t*18)
# tear plinks / water ball wobbles
for at in (0.3,1.6,5.0,8.3,10.5,13.4,14.8): add(plink(rng.uniform(1400,2200)),at,0.2)
for at in np.arange(13.5,16.8,0.55):
    tw=tt(0.3); add(np.sin(2*np.pi*(300+80*np.sin(2*np.pi*12*tw))*tw)*np.exp(-tw*8),at,0.12)
# squirt + floating droplets
s=rng.normal(size=int(SR*0.35)); s=np.convolve(s,np.ones(6)/6,'same')*np.exp(-tt(0.35)*8); add(s,17.3,0.35)
for at in np.arange(17.6,23.8,0.6): add(plink(rng.uniform(1000,1600),0.18),at,0.12)
# label pops
for at in (0.15,3.2,4.0,7.7,10.2,13.4,17.2,20.5,24.2,25.4):
    s=rng.normal(size=int(SR*0.06)); s=np.convolve(s,np.ones(8)/8,'same')*np.exp(-tt(0.06)*60); add(s,at,0.28)
for at in (2.9,9.9,16.9,23.9,27.1):
    d=0.4; s=rng.normal(size=int(SR*d)); s=np.convolve(s,np.ones(40)/40,'same')*np.sin(np.pi*tt(d)/d); add(s,at,0.6)
# comedic 'wah' at peçete
t2=tt(0.6); add(np.sign(np.sin(2*np.pi*np.cumsum(330-140*t2)/SR))*np.exp(-t2*4)*0.3,25.4,0.15)
for f,g in ((1318,1),(1760,0.6),(2637,0.3)): add(np.sin(2*np.pi*f*tt(1.6))*np.exp(-tt(1.6)*2.5),27.2,0.08*g)
mix/=np.abs(mix).max()/0.85; mix[-int(SR*0.6):]*=np.linspace(1,0,int(SR*0.6))
w=wave.open('aud.wav','wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix*32767).astype(np.int16).tobytes()); w.close()
