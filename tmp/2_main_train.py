import torch
# lets do som predictions
import tiktoken

from dg_lib2 import HF_GPT2, GPTConfig
from dg_lib2 import DataLoader, Trainer,Trainer_base

#params 
B,T=8,1024 #minibatch and legth 
batch_size=2**19 #batch size

B,T=4,1024 #minibatch and legth 
batch_size=4*1024*2 #batch size


#scheduler
lr_max=3e-4 
lr_min=lr_max*0.1
warmup_iters=10
lr_decay_iters=20

#iters
steps=1000

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
trainer=Trainer()
trainer.batch_size=batch_size
trainer.lr_scheduler.max_lr=lr_max
trainer.lr_scheduler.min_lr=lr_min
trainer.lr_scheduler.warmup_iters=warmup_iters
trainer.lr_scheduler.lr_decay_iters=lr_decay_iters

torch.compile()

#train
trainer.load_checkpoint('../../checkpoints/cp1.cp')
trainer.train(model,dl,steps)
trainer.save_checkpoint('../../checkpoints/cp1.cp')
#trainer.train(model,dl,steps)
#trainer.save_checkpoint('../../checkpoints/cp1.cp')
#save
model.save_safetensor(("../../checkpoints/dg2.safetensors"))



##

#trainer=Trainer_base()

#torch.compile()

#train
#trainer.train(model,dl,steps)



#model.save_statedic("./checkpoint.safetensors")

#uncomment to check initial loss
#logits,loss=model(x,y)
#print(loss)