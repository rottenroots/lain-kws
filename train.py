import os, glob, random, subprocess, tempfile
import numpy as np, soundfile as sf, torch, torch.nn as nn
from math import gcd
from scipy.signal import resample_poly, butter, sosfilt
random.seed(1); np.random.seed(1); torch.manual_seed(1)
SR, WIN = 16000, 32000
VOICES = glob.glob('voices/*.onnx')
POS = ["Lain", "Láin"]
NEGW = ["lein","lei","lã","Laila","ali","plain","lane","lá","laço","lanche","Lúcia","vai","Lauro",
        "alô","ai","aí","Laís","Lia","Lara","Lina","rain","pain","ok","olha","Bolsonaro","main","gain","Elaine","lenha","linha","nada","claro"]
NEGS = ["Aí eu falei para ele","Laís chegou cedo","Aline foi embora","Lara e Lia saíram","Ainda não terminei","Maior que isso não tem","Que horas são agora","Abre o aplicativo de música","Eu vou sair daqui a pouco",
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

def raw(chunk, gain=None):
    x = chunk.astype(np.float32)
    if len(x) < WIN: x = np.pad(x, (0, WIN-len(x)))
    x = x[:WIN]
    x = x * (gain if gain else random.uniform(0.3, 3.0))
    x = x + colored(WIN, random.choice([0,1,2])) * random.uniform(0, 0.01)
    if random.random() < 0.3: x = sosfilt(HFP, x)
    return (np.clip(x,-1,1)*32767).astype(np.int16)

def place(word, ctx=None):
    w = word[:WIN-4000]
    base = np.zeros(WIN, np.float32) if ctx is None else ctx.copy()
    end = random.randint(max(int(WIN*0.7), len(w)), WIN-800)
    end = max(end, len(w))
    base[end-len(w):end] = w
    return base

def windows(x, hop=8000):
    return [x[i:i+WIN] for i in range(0, len(x)-WIN+1, hop)]

def synth_clip(w):
    return w/(np.abs(w).max()+1e-9)*random.uniform(0.05, 0.4)

X, Y, G = [], [], []
def add(a, y, g): X.append(a); Y.append(y); G.append(g)

for v in VOICES:
    for t in POS:
        for ls in [0.8,0.9,1.0,1.1,1.25]:
            for ns in [0.4,0.9]:
                w = tts(t, v, ls, ns)
                if w is None or len(w) < 2000: continue
                for _ in range(6): add(raw(place(synth_clip(w))), 1, 'synth')
    for t in NEGW+NEGS:
        for ls in [0.9,1.0,1.15]:
            w = tts(t, v, ls, 0.667)
            if w is None: continue
            for _ in range(4): add(raw(place(synth_clip(w))), 0, 'synth')
for _ in range(200): add(raw(np.zeros(WIN,np.float32)), 0, 'synth')

negx = readwav('data/neg_longo.wav') if os.path.exists('data/neg_longo.wav') else np.zeros(WIN*4, np.float32)
cut = int(len(negx)*0.75)
def bg():
    i = random.randint(0, cut-WIN); return negx[i:i+WIN]*random.uniform(0.0, 0.5)
for fn, g, rep in [('data/lain_voz.wav','voz',40), ('data/t_lain.wav','t_lain',30)]:
    if not os.path.exists(fn): continue
    S = segs(readwav(fn)); print(fn, "segmentos:", len(S))
    for w in S:
        for _ in range(rep): add(raw(place(w, bg())), 1, g)

for w in windows(negx[:cut], 4000): add(raw(w), 0, 'neglong_train')
for w in windows(negx[cut:], 8000): add(raw(w), 0, 'neglong_hold')
if os.path.exists('data/t_neg.wav'):
    for w in windows(readwav('data/t_neg.wav'), 8000): add(raw(w), 0, 't_neg')

X = np.stack(X); Y = np.array(Y, np.float32); G = np.array(G)
for g in sorted(set(G)): print(f"grupo {g}: {(G==g).sum()} (pos={int(Y[G==g].sum())})")
from openwakeword.utils import AudioFeatures
F = AudioFeatures(inference_framework='onnx')
E = F.embed_clips(X, batch_size=64).astype(np.float32)
T = E.shape[1]; print("embeddings:", E.shape)

def fit(idx, epochs=60):
    net = nn.Sequential(nn.Flatten(), nn.Linear(T*96,64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64,1), nn.Sigmoid())
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    Et, Yt = torch.tensor(E), torch.tensor(Y)
    P = idx[Y[idx]==1]; N = idx[Y[idx]==0]
    for ep in range(epochs):
        net.train(); np.random.shuffle(N); L = []
        for i in range(0, len(N), 64):
            nb = N[i:i+64]; pb = np.random.choice(P, len(nb))
            b = np.concatenate([nb, pb])
            loss = nn.functional.binary_cross_entropy(net(Et[b]).squeeze(1), Yt[b])
            opt.zero_grad(); loss.backward(); opt.step(); L.append(loss.item())
        if ep % 10 == 9 or ep == 0:
            net.eval()
            with torch.no_grad(): o = net(Et[idx]).squeeze(1).numpy()
            print(f"ep {ep+1}: loss={np.mean(L):.4f} saida media={o.mean():.3f} desvio={o.std():.3f}")
            net.train()
    net.eval()
    with torch.no_grad(): o = net(Et[idx]).squeeze(1).numpy()
    if o.std() < 0.1: raise SystemExit("MODELO COLAPSADO (desvio da saida < 0.1)")
    return net
def score(net, ids):
    with torch.no_grad(): return net(torch.tensor(E[ids])).squeeze(1).numpy()

tr = np.where(np.isin(G, ['synth','voz','neglong_train']))[0]
net = fit(tr)
lines = [f"T={T}"]
pos = np.where(G=='t_lain')[0]; pp = score(net, pos)
for name in ['t_neg','neglong_hold']:
    pn = score(net, np.where(G==name)[0])
    for th in [0.5,0.8,0.9,0.95,0.99]:
        lines.append(f"[{name}] limiar {th}: recall(t_lain)={(pp>th).mean():.3f} falso_positivo/janela={(pn>th).mean():.4f}")
print("\n".join(lines)); open('out/metrics.txt','w').write("\n".join(lines))
torch.onnx.export(net, torch.zeros(1,T,96), 'out/lain_head.onnx', input_names=['emb'],
                  output_names=['p'], opset_version=13, dynamo=False)
print("ok final")
