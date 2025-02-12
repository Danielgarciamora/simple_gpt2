from datasets import load_dataset

import pickle
import gzip

from torch.utils.data import Dataset, DataLoader


import torch
# lets do som predictions
import tiktoken

from dg_lib2 import HF_GPT2, GPTConfig
from dg_lib2 import DataLoader, Trainer,Trainer_base,FineWebDataset



a=FineWebDataset()
#a.download_records(1e6)
#records=a.load_local()
#records[0]
B,T=8,1024 #minibatch and legth 
dl=DataLoader(B,T)
dl.load_local_fineweb("../../data/fineweb/fineweb_records_1000000.0_0.pkl.gz")
#dl.save_tokens_indices("../../data/fineweb/fineweb_records_1000000.0_0.pkl.gz.idx")

