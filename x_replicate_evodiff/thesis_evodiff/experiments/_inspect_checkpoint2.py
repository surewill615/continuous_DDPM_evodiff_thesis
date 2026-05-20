import sys; sys.path.insert(0, '.')
from dnnlib.util import open_url
import pickle, torch
import numpy as np

f = open_url('checkpoints/edm-cifar10-32x32-uncond-ve.pkl')
data = pickle.load(f)

ema = data['ema']

# Try to see if ema is callable
print("EMA callable:", callable(ema))
print("EMA dir:", [x for x in dir(ema) if not x.startswith('_')][:30])

# Check for common EDM method names
for attr_name in ['forward', '__call__', 'F', 'G', 'H', 'num_blocks']:
    if hasattr(ema, attr_name):
        print(f"  has {attr_name}:", type(getattr(ema, attr_name)))

# Try forward pass with a dummy image
try:
    device = 'cpu'
    x = torch.randn(1, 3, 32, 32)
    sigma = torch.ones(1) * 0.5
    with torch.no_grad():
        # Try callable
        out = ema(x, sigma)
        print(f"ema(x, sigma) output shape: {out.shape}")
        print(f"ema(x, sigma) output range: [{out.min():.4f}, {out.max():.4f}]")
except Exception as e:
    print(f"ema(x, sigma) failed: {e}")

# Try with class labels
try:
    x = torch.randn(1, 3, 32, 32)
    sigma = torch.ones(1) * 0.5
    class_labels = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        out = ema(x, sigma, class_labels)
        print(f"ema(x, sigma, class_labels) output shape: {out.shape}")
except Exception as e:
    print(f"ema(x, sigma, class_labels) failed: {e}")

# Check augment_pipe
print("\naugment_pipe:", type(data['augment_pipe']))
if data['augment_pipe'] is not None:
    print("augment_pipe dir:", [x for x in dir(data['augment_pipe']) if not x.startswith('_')][:20])