import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
print('CUDA available:', torch.cuda.is_available())

import torchvision.models as models
m = models.inception_v3(weights='IMAGENET1K_V1', transform_input=False)
print('Inception V3 loaded OK:', m.__class__.__name__)