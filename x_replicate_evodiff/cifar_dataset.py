from datasets import load_dataset

ds = load_dataset("uoft-cs/cifar10")

custom_cache_dir = "/home/yuantao/code/xwy" 

ds = load_dataset("uoft-cs/cifar10", cache_dir=custom_cache_dir)