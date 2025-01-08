import torch
# lets do som predictions
import tiktoken

from dg_lib2 import HF_GPT2



torch.manual_seed(0)
torch.cuda.manual_seed(0)


num_return_sequences=2
max_length=30
prompt="hello my name is"


model=HF_GPT2.from_pretrained('gpt2')
model.to("cuda")
responses=model.generate(prompt,max_length,num_return_sequences)
for r in responses:
    print(">"+r)