import os, glob, random, subprocess, tempfile
import numpy as np, soundfile as sf, torch, torch.nn as nn
from math import gcd
from scipy.signal import resample_poly, butter, sosfilt
random.seed(1); np.random.seed(1); torch.manual_seed(1)
SR, WIN = 16000, 32000
VOICES = glob.glob('voices/*.onnx')
POS = ["Lain", "Láin"]
NEGW = ["lein","lei","lã","Laila","ali","plain","lane","lá","laço","lanche","Lúcia","vai","Lauro",
        "alô","ok","olha","Bolsonaro","main","gain","Elaine","lenha","linha","nada","claro"]
NEGS = ["Que horas são agora","Abre o aplicativo de música","Eu vou sair daqui a pouco",
        "Liga a luz da sala","Não sei o que fazer hoje","Me lembra de comprar pão"]
os.makedirs('out', exist_ok=True)

def tts(text, voice, ls, ns):
    out = tempfile.mktemp(suffix='.wav')
    r = subprocess.run(['piper','-m',voice,'-f',out,'--length_scale',str(ls),'--noise_scale',str(ns)],
                       input=text.encode(), capture_output=True)
    if r.returncode or not os.path.exists(out): return None
    x, sr = sf.read(out, dtype='float32'); os.remove(out)
    if x.ndim > 1: x = x[:, 0]
    g = gcd(SR, sr); x = resample_poly(x, SR//g, sr//g).astype(np.float32)
    idx = np.where(np.abs(x) > 0.02*np.abs(x).max())[0]
    return x[idx[0]:idx[-1]+1] if len(idx) else None

def colored(n, beta):
    S = np.fft.rfft(np.random.randn(n)); f = np.fft.rfftfreq(n); S[1:] /= f[1:]**(beta/2)
    x = np.fft.irfft(S, n); return (x/(x.std()+1e-9)).astype(np.float32)

HFP = butter(4, [300, 3400], 'bandpass', fs=SR, output='sos')
def make(word, positive):
    sig = np.zeros(WIN, np.float32)
    if word is not None:
        w = word[:WIN-4000]; w = w/(np.abs(w).max()+1e-9)*random.uniform(0.5, 1.0)
        end = random.randint(int(WIN*0.7), WIN-800) if positive else random.randint(len(w), WIN)
        sig[end-len(w):end] = w
        lvl = random.uniform(0.01, 0.3)   # ruido relativo (SNR ~10-40 dB)
    else:
        lvl = random.uniform(0.0005, 0.5)
    x = sig + colored(WIN, random.choice([0,1,2]))*lvl*0.3
    if random.random() < 0.5: x = sosfilt(HFP, x)
    x = x/(np.abs(x).max()+1e-9)*random.uniform(0.2, 0.9)
    return (np.clip(x,-1,1)*32767).astype(np.int16)

def segs(x, minl=0.15, maxl=1.2):
    n = len(x)//160; e = 10*np.log10((x[:n*160].reshape(n,160)**2).mean(1)+1e-10)
    act = e > np.percentile(e,10)+12; out=[]; s=None; h=0
    for i,a in enumerate(act):
        if a:
            if s is None: s=i
            last=i; h=20
        elif s is not None:
            h-=1
            if h<=0: out.append((s,last+1)); s=None
    if s is not None: out.append((s,last+1))
    return [x[max(0,a-5)*160:(b+5)*160] for a,b in out if minl <= (b-a)*0.01 <= maxl]

def readwav(p):
    x, sr = sf.read(p, dtype='float32'); assert sr == SR; return x if x.ndim==1 else x[:,0]

X, Y = [], []
# positivos sinteticos
n=0
for v in VOICES:
    for t in POS:
        for ls in [0.8,0.9,1.0,1.1,1.25]:
            for ns in [0.4,0.9]:
                w = tts(t, v, ls, ns)
                if w is None or len(w) < 2000: continue
                n+=1
                for _ in range(16): X.append(make(w,True)); Y.append(1)
print("positivos TTS:", n)
# positivos: sua voz
if os.path.exists('data/lain_voz.wav'):
    S = segs(readwav('data/lain_voz.wav'))
    print("segmentos da sua voz:", len(S))
    for w in S:
        for _ in range(40): X.append(make(w,True)); Y.append(1)
# negativos TTS
n=0
for v in VOICES:
    for t in NEGW+NEGS:
        for ls in [0.9,1.0,1.15]:
            w = tts(t, v, ls, 0.667)
            if w is None: continue
            n+=1
            for _ in range(12): X.append(make(w,False)); Y.append(0)
print("negativos TTS:", n)
# negativos: sua voz
if os.path.exists('data/neg_voz.wav'):
    x = readwav('data/neg_voz.wav'); S = segs(x, 0.1, 1.9)
    print("segmentos neg sua voz:", len(S))
    for w in S:
        for _ in range(30): X.append(make(w,False)); Y.append(0)
# so ruido / silencio
for _ in range(500): X.append(make(None,False)); Y.append(0)

X = np.stack(X); Y = np.array(Y, np.float32)
print("total:", len(X), "pos:", int(Y.sum()))
from openwakeword.utils import AudioFeatures
F = AudioFeatures(inference_framework='onnx')
E = F.embed_clips(X, batch_size=64).astype(np.float32)
T = E.shape[1]; print("embeddings:", E.shape, "T =", T)

idx = np.random.permutation(len(E)); k = int(len(E)*0.8)
tr, va = idx[:k], idx[k:]
net = nn.Sequential(nn.Flatten(), nn.Linear(T*96,64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64,1), nn.Sigmoid())
opt = torch.optim.Adam(net.parameters(), 1e-3)
Et, Yt = torch.tensor(E), torch.tensor(Y)
wpos = float(min((Y==0).sum()/max(1,Y.sum()), 3))
for ep in range(120):
    net.train(); p = np.random.permutation(tr)
    for i in range(0, len(p), 128):
        b = p[i:i+128]; o = net(Et[b]).squeeze(1)
        w = torch.where(Yt[b]==1, torch.tensor(wpos), torch.tensor(1.0))
        loss = nn.functional.binary_cross_entropy(o, Yt[b], weight=w)
        opt.zero_grad(); loss.backward(); opt.step()
net.eval()
with torch.no_grad(): pv = net(Et[va]).squeeze(1).numpy()
yv = Y[va]; res = []
for th in [0.5,0.8,0.9,0.95]:
    rec = (pv[yv==1]>th).mean(); fp = (pv[yv==0]>th).mean()
    res.append(f"limiar {th}: recall={rec:.3f} falso_positivo={fp:.4f}")
print("\n".join(res)); open('out/metrics.txt','w').write(f"T={T}\n"+"\n".join(res))
torch.onnx.export(net, torch.zeros(1,T,96), 'out/lain_head.onnx', input_names=['emb'],
                  output_names=['p'], opset_version=13, dynamo=False)
print("ok")
