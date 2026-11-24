import numpy as np, wave
SR=44100; D=30.0; N=int(SR*D); mix=np.zeros(N); rng=np.random.default_rng(4)
def tt(d): return np.arange(int(SR*d))/SR
def add(s,at,g=1.):
    i=int(at*SR)
    if i>=N: return
    j=min(N,i+len(s)); mix[i:j]+=s[:j-i]*g
def nf(m): return 440*2**((m-69)/12)
def marimba(f,d=0.9):
    t=tt(d); return (np.sin(2*np.pi*f*t)+0.35*np.sin(2*np.pi*f*4*t)*np.exp(-t*20))*np.exp(-t*5)
def bass(f,d=0.6):
    t=tt(d); return np.sin(2*np.pi*f*t)*np.exp(-t*4)
# playful mysterious waltz-ish (3/4, 100bpm) in A minor -> lighter on end
beat=60/100
prog=[(57,60,64),(53,57,60),(55,59,62),(52,56,59)]
b=0
while b*beat<D-0.5:
    at=b*beat; ch=prog[(b//6)%4]; pos=b%3
    g=1.0
    if 3.0<at<3.3 or 10.8<at<11.1 or 17.3<at<17.6: g=0.4
    if pos==0: add(bass(nf(ch[0]-24)),at,0.18*g)
    add(marimba(nf(ch[(b+pos)%3]+12)),at,0.07*g)
    if b%6==5: add(marimba(nf(ch[2]+24),0.6),at+beat/2,0.04*g)
    b+=1
# geiger clicks
def click():
    c=tt(0.004); return np.sin(2*np.pi*3200*c)*np.exp(-c*2000)+rng.normal(size=len(c))*np.exp(-c*1500)*0.5
def geiger(a,bb,rate_fn,g):
    tc=a
    while tc<bb:
        tc+=rng.exponential(1/max(0.5,rate_fn(tc)))
        if tc<bb: add(click(),tc,g)
geiger(0,3.2,lambda t:6,0.35)
geiger(11.0,17.5,lambda t:3+ (min(max(t-12.4,0),1.4)/1.4)*55,0.4)
geiger(17.5,25,lambda t:2,0.2)
# label pops (paper)
for at in (0.15,3.4,4.4,7.3,9.25,11.2,13.6,15.7,17.7,19.4,21.5,23.0,25.15,25.6,26.5):
    s=rng.normal(size=int(SR*0.06)); s=np.convolve(s,np.ones(8)/8,'same')*np.exp(-tt(0.06)*60); add(s,at,0.35)
# whooshes at cuts
for at in (3.1,10.9,17.4,24.9):
    d=0.4; s=rng.normal(size=int(SR*d)); s=np.convolve(s,np.ones(40)/40,'same')*np.sin(np.pi*tt(d)/d); add(s,at,0.8)
# magical shimmer when flask glows / blink
for at in (7.3,9.5):
    for k,f in enumerate((1568,1976,2349,2637)):
        add(np.sin(2*np.pi*f*tt(0.8))*np.exp(-tt(0.8)*5),at+k*0.05,0.03)
# stamp/thud for 'Defterlerine bile'
t=tt(0.4); add(np.sin(2*np.pi*np.cumsum(45+90*np.exp(-t*25))/SR)*np.exp(-t*9),13.6,0.5)
# gulp for visitor
t=tt(0.25); add(np.sin(2*np.pi*np.cumsum(500-300*t/0.25)/SR)*np.exp(-t*12),23.0,0.25)
# end ding
for f,g in ((1318,1),(1760,0.6),(2637,0.3)): add(np.sin(2*np.pi*f*tt(1.6))*np.exp(-tt(1.6)*2.5),27.2,0.08*g)
mix/=np.abs(mix).max()/0.85; mix[-int(SR*0.6):]*=np.linspace(1,0,int(SR*0.6))
w=wave.open('aud.wav','wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix*32767).astype(np.int16).tobytes()); w.close()
