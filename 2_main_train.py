import torch
# lets do som predictions
import tiktoken

from dg_lib2 import HF_GPT2, GPTConfig
from dg_lib2 import DataLoader
enc=tiktoken.get_encoding('gpt2')


B,T=4,32

dl=DataLoader(B,T)
data=dl.load_txt('./data/tinyshakespeare/input.txt')
#tokens=dl.next_batch()

config=GPTConfig()
model=HF_GPT2(config)



#uncomment to check initial loss
#logits,loss=model(x,y)
#print(loss)

#To cuda
model.to('cuda')

torch.compile()

#optimize
lr=3e-4
optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
for i in range(50):
    
    x,y=dl.next_batch()
    x=x.to('cuda')
    y=y.to('cuda')  

    optimizer.zero_grad()
    logits,loss=model(x,y)
    loss.backward()
    optimizer.step()
    print(f"step {i}, loss: {loss.item()}")