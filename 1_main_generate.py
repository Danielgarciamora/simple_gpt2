import torch
# lets do som predictions
import tiktoken

from dg_lib2 import HF_GPT2,GPTConfig



torch.manual_seed(0)
torch.cuda.manual_seed(0)


num_return_sequences=2
max_length=100
prompt="person0: Hello? \n person1: "

#Load from HF
#model=HF_GPT2.from_pretrained('gpt2')
#save
#model.save_safetensor("../../checkpoints/gpt2.safetensors")

#load local file
config=GPTConfig()
config.n_layer=12
model=HF_GPT2(config)           
#model.load_safetensor("../../checkpoints/gpt2.safetensors")
model.load_safetensor("../../checkpoints/dg2.safetensors")


#generate
model.to("cuda")
responses=model.generate(prompt,max_length,num_return_sequences)
for r in responses:
    print(">"+r)

