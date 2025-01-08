import torch
# lets do som predictions
import tiktoken

from dg_lib2 import HF_GPT2

enc=tiktoken.get_encoding('gpt2')

torch.manual_seed(42)
torch.cuda.manual_seed(42)

num_return_sequences=5
max_length=30

tokens=enc.encode("hello my name is")
#list to tensor
tokens=torch.tensor(tokens,dtype=torch.long)
tokens=tokens.unsqueeze(0).repeat(num_return_sequences,1)

model=HF_GPT2.from_pretrained('gpt2')
x=tokens

#move model and input to gpu
model.to('cuda') 
x=tokens.to('cuda') 

#generate
a=model.generate(x,max_length)

#decode 
for i in range(num_return_sequences):
    tokens=a[i,:max_length].tolist()
    decoded=enc.decode(tokens)
    print(">",decoded)
#--------------

#print("----------")
#from train_gpt2_2 import GPT
#model2=GPT.from_pretrained('gpt2')
#torch.manual_seed(42)
#torch.cuda.manual_seed(42)
#a=model2.generate(x,max_length)
#
#for i in range(num_return_sequences):
#    tokens=a[i,:max_length].tolist()
#    decoded=enc.decode(tokens)
#    print(">",decoded)
#model2.state_dict()
#
#
#model2.state_dict()[list(model2.state_dict().keys())[0]]
#model.state_dict()[list(model.state_dict().keys())[0]]
#xxx
#    
#
#
#
#
#
#
##example
#a=torch.tensor([[1,2],[3,4]])
#
#
#b_size=1
#seq_l=4
#n_embd=2
#n_heads=1
#
#mha=MultiHeadAttention(n_embd,n_heads)
#q=torch.randn(b_size,seq_l,n_embd)
#k=torch.randn(b_size,seq_l,n_embd)
#v=torch.randn(b_size,seq_l,n_embd)
#
#mask=torch.ones(b_size,1,1,seq_l)
#
#out,att_w=mha(q,k,v,mask)



