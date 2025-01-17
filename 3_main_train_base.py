import torch
# lets do som predictions
import tiktoken

from dg_lib2 import HF_GPT2, GPTConfig
from dg_lib2 import DataLoader, Trainer,Trainer_base

#params 
B,T=4,32 #minibatch and legth 


#iters
steps=50

#dataloader
dl=DataLoader(B,T)
data=dl.load_txt('../../data/tinyshakespeare/input.txt')
#tokens=dl.next_batch()

#model config
config=GPTConfig()
config.n_layer=12

#model
model=HF_GPT2(config)

#To cuda
model.to('cuda')

#trainer
trainer=Trainer_base()

torch.compile()

#train
trainer.train(model,dl,steps)

#save
model.save_safetensor(("../../checkpoints/dg.safetensors"))

#model.save_statedic("./checkpoint.safetensors")

#uncomment to check initial loss
#logits,loss=model(x,y)
#print(loss)