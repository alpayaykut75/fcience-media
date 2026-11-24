import numpy as np, wave
SR=44100; D=30.0; N=int(SR*D); mix=np.zeros(N); rng=np.random.default_rng(8)
def tt(d): return np.arange(int(SR*d))/SR
def add(s,at,g=1.):
    i=int(at*SR)
    if i>=N: return
    j=min(N,i+len(s)); mix[i:j]+=s[:j-i]*g
def nf(m): return 440*2**((m-69)/12)
def marimba(f,d=0.8):
    t=tt(d); return (np.sin(2*np.pi*f*t)+0.35*np.sin(2*np.pi*f*4*t)*np.exp(-t*20))*np.exp(-t*5)
def bass(f,d=0.5): t=tt(d); return np.sin(2*np.pi*f*t)*np.exp(-t*4)
beat=60/116; prog=[(60,64,67),(65,69,72),(62,65,69),(67,71,74)]
b=0
while b*beat<D-0.5:
    at=b*beat; ch=prog[(b//4)%4]; g=1.0
    if any(c-0.2<at<c+0.1 for c in (3.0,9.5,15.5,21.0)): g=0.4
    if b%2==0: add(bass(nf(ch[0]-24)),at,0.17*g)
    add(marimba(nf(ch[b%3]+12)),at,0.065*g)
    if b%4==3: add(marimba(nf(ch[2]+24),0.5),at+beat/2,0.035*g)
    b+=1
def blip(f,d=0.06): t=tt(d); return np.sin(2*np.pi*f*t)*np.exp(-t*50)
# radar pings
for at in np.arange(0.5,21,2.6):
    t=tt(0.7); add(np.sin(2*np.pi*1400*t)*np.exp(-t*7),at,0.05)
# chocolate drip blops
for at in (0.6,1.4,2.2,7.0,7.8):
    t=tt(0.12); f=600-2500*t; add(np.sin(2*np.pi*np.cumsum(f)/SR)*np.exp(-t*25),at,0.25)
# label paper pops
for at in (0.15,3.2,4.1,6.7,9.7,12.5,15.6,18.3,21.2,24.3):
    s=rng.normal(size=int(SR*0.06)); s=np.convolve(s,np.ones(8)/8,'same')*np.exp(-tt(0.06)*60); add(s,at,0.3)
# whooshes at cuts
for at in (2.9,9.4,15.4,20.9,27.1):
    d=0.4; s=rng.normal(size=int(SR*d)); s=np.convolve(s,np.ones(40)/40,'same')*np.sin(np.pi*tt(d)/d); add(s,at,0.7)
# popcorn pops
tc=9.9
while tc<15.3:
    tc+=rng.exponential(0.09 if tc<13 else 0.2)
    t=tt(0.03); add(rng.normal(size=len(t))*np.exp(-t*200)+np.sin(2*np.pi*rng.uniform(800,1600)*t)*np.exp(-t*150),tc,rng.uniform(0.25,0.5))
# egg: tension rise then splat
t=tt(1.1); add(np.sin(2*np.pi*np.cumsum(200+600*t)/SR)*0.5*t,15.7,0.15)
t=tt(0.5); add((rng.normal(size=len(t))*np.exp(-t*12))*0.8+np.sin(2*np.pi*np.cumsum(90*np.exp(-t*6)+40)/SR)*np.exp(-t*8),16.8,0.9)
for k in range(2):
    t=tt(0.4); f=220+500*(1-np.exp(-t*8)); add(np.sin(2*np.pi*np.cumsum(f)/SR)*np.exp(-t*5),16.95+k*0.12,0.25)
# microwave hum + ding
t=tt(5.5); add((np.sin(2*np.pi*120*t)+0.3*np.sin(2*np.pi*240*t))*0.4,21.2,0.06)
for f,g in ((1568,1),(2093,0.5)): add(np.sin(2*np.pi*f*tt(1.4))*np.exp(-tt(1.4)*3),24.3,0.12*g)
# end ding
for f,g in ((1318,1),(1760,0.6),(2637,0.3)): add(np.sin(2*np.pi*f*tt(1.6))*np.exp(-tt(1.6)*2.5),27.2,0.08*g)
mix/=np.abs(mix).max()/0.85; mix[-int(SR*0.6):]*=np.linspace(1,0,int(SR*0.6))
w=wave.open('aud.wav','wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix*32767).astype(np.int16).tobytes()); w.close()
