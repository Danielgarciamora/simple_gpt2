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
a.download_records(1e6)
#records=a.load_local()
#records[0]
