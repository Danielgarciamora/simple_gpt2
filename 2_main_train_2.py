import torch
# lets do som predictions
import tiktoken

from dg_lib2 import HF_GPT2, GPTConfig
from dg_lib2 import DataLoader, Trainer,Trainer_base

#params 
B,T=8,1024 #minibatch and legth 
batch_size=2**19 #batch size

B,T=4,1024 #minibatch and legth 
#batch_size=4*1024*32 #batch size
batch_size=2**19 #batch size


#scheduler
#gpt 3
#tokens 300b
#decay=260b
#warm 0.375b
#lr=6e-4

lr_max=6e-4 
lr_min=lr_max*0.1
warmup_iters=10
lr_decay_iters=100

#iters
steps=160

#dataloader
dl=DataLoader(B,T)
dl.load_txt("../../data/dic/Oxford English Dictionary.txt")
dl.load_local_fineweb('../../data/fineweb/fineweb_records_1000000.0_0.pkl.gz')
#dl.load_local_fineweb('../../data/fineweb/fineweb_records_1000000.0_1.pkl.gz')
dl.load_dailydialog('../../data/dailydialog/dialogues_train.txt')
dl.rand_split()
dl.encode() 

#dl.save_tokens('train_tokens.tk')

print("a")

#data=dl.load_txt('../../data/dailydialog/dialogues_train.txt')
#data=dl.load_dailydialog('../../data/dailydialog/dialogues_train.txt')
#data=dl.load_txt('../../data/dic/Oxford English Dictionary.txt')


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


torch.set_float32_matmul_precision('high')
torch.compile()
torch.manual_seed(0)
torch.cuda.manual_seed(0)

#train
#trainer.load_checkpoint('../../checkpoints/cp2.cp')
print(f"tokens: {dl.get_train_token_lenght()}")

print(f"estimated train duration: {steps*batch_size/2000/60/60} H")



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
#print(loss)- -. 