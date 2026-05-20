import sys; sys.path.insert(0, '.')
from dnnlib.util import open_url
import pickle, torch

f = open_url('checkpoints/edm-cifar10-32x32-uncond-ve.pkl')
data = pickle.load(f)

print("Top-level keys:", list(data.keys()))
ema = data['ema']
print("EMA type:", type(ema))
if isinstance(ema, dict):
    print("EMA is dict, sample keys:", list(ema.keys())[:10])
elif hasattr(ema, '__dict__'):
    print("EMA attrs:", dir(ema)[:20])

# Check dataset_kwargs
print("\ndataset_kwargs:", data.get('dataset_kwargs', 'N/A'))

# Check loss_fn
print("loss_fn:", type(data.get('loss_fn', 'N/A')))