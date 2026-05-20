import sys; sys.path.insert(0, '.')
import importlib

mods = ['torch', 'torchvision', 'numpy', 'tqdm', 'pytorch_fid', 'cleanfid', 'piq']
for m in mods:
    try:
        importlib.import_module(m)
        print(f"{m}: OK")
    except:
        print(f"{m}: MISSING")

# Also check computed metrics module
try:
    from utils.metrics import compute_fid
    print("utils.metrics.compute_fid: OK")
except Exception as e:
    print(f"utils.metrics.compute_fid: {e}")

# Check EDM specific - what's the noise schedule?
from dnnlib.util import open_url
import pickle
f = open_url('checkpoints/edm-cifar10-32x32-uncond-ve.pkl')
data = pickle.load(f)
ema = data['ema']
print(f"\nEMA img_channels: {ema.img_channels}")
print(f"EMA img_resolution: {ema.img_resolution}")
print(f"EMA label_dim: {ema.label_dim}")

# Check if KarrasHeun from EDM repo is available
try:
    sys.path.insert(0, 'C:/Users/will/Desktop/thesis_edm')  # common EDM repo location
    from edm import KarrasHeunDenoiser
    print("KarrasHeunDenoiser from edm repo: OK")
except:
    print("KarrasHeunDenoiser not found in ~/edm")

# Check what the score model actually returns
import torch
device = 'cpu'
x = torch.randn(2, 3, 32, 32)
sigma = torch.ones(2) * 0.5
with torch.no_grad():
    print(f"\nema.forward type: {type(ema.forward)}")
    out1 = ema.forward(x, sigma)
    print(f"forward(x, sigma) shape: {out1.shape}")
    # Try as denoiser
    out2 = ema(x, sigma)
    print(f"__call__(x, sigma) shape: {out2.shape}")
    print(f"Difference forward vs __call__: {(out1-out2).abs().max():.6f}")