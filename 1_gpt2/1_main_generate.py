import torch
# lets do som predictions
import tiktoken

from dg_lib2 import HF_GPT2,GPTConfig



torch.manual_seed(0)
torch.cuda.manual_seed(0)


num_return_sequences=3
max_length=50
#prompt="user:Hello, nice to meet you!, who are you?\AI:"
#prompt="Person0:what do you think about computers?\nPerson1:"
#prompt="Person0:what's the best sport?\nPerson1:"
#prompt="Person0:which is the bigest country?\nPerson1:"
prompt="I'm an artificial intelligence"

#Load from HF
#model=HF_GPT2.from_pretrained('gpt2')
#save
#model.save_safetensor("../../checkpoints/gpt2.safetensors")

#load local file
config=GPTConfig()
config.n_layer=12
model=HF_GPT2(config)           
model.load_safetensor("../../checkpoints/gpt2.safetensors")
#model.load_safetensor("../../checkpoints/dg2.safetensors")

params=model.get_num_params(False)
print(f"num params:{params}")

#generate
model.to("cuda")
responses=model.generate(prompt,max_length,num_return_sequences)
for r in responses:
    print(">"+r)
    print("\n")
    print("\n")

