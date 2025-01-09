import torch
# lets do som predictions
import tiktoken

from dg_lib2 import HF_GPT2, GPTConfig
from dg_lib2 import DataLoader, Trainer

B,T=4,32

dl=DataLoader(B,T)
data=dl.load_txt('../../data/tinyshakespeare/input.txt')
#tokens=dl.next_batch()

config=GPTConfig()
config.n_layer=12
model=HF_GPT2(config)

#model.save_statedic("./checkpoint.safetensors")

#uncomment to check initial loss
#logits,loss=model(x,y)
#print(loss)

#To cuda
model.to('cuda')

torch.compile()

trainer=Trainer()

trainer.train(model,dl,5000)

model.save_safetensor(("../../checkpoints/dg.safetensors"))
