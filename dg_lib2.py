import torch.nn as nn
import torch
import math

from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size:int=50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    block_size: int = 1024
    n_layer:int=128
    n_head:int=12
    n_embd: int=768
    dropout:float=0
    bias:bool=True

class MultiHeadAttention(nn.Module):
    def __init__(self,config:GPTConfig):
        
        n_embd=config.n_embd
        n_heads=config.n_head
        super().__init__()
        
        #dimensionality of each head
        self.n_head_dim=n_embd // n_heads
        self.n_heads=n_heads
        self.n_embd=n_embd
        
        #Linear
        self.w_q=nn.Linear(n_embd,n_embd)
        self.w_k=nn.Linear(n_embd,n_embd)
        self.w_v=nn.Linear(n_embd,n_embd)
        
        
        #final layer
        self.c_proj=nn.Linear(n_embd,n_embd)
        
        #scale to stabilize gradient
        self.scale=math.sqrt(self.n_head_dim)
        
    def forward(self,q,k,v, mask=None,torch_attention=False):
        
        b_size, seq_l, n_embd = q.size() # batch size, sequence length, embedding dimensionality (n_embd)

        #linear proj
        q=self.w_q(q)
        k=self.w_k(k)
        v=self.w_k(v)
        
        #reshape from (B,L,D) to (B,L,H,HD) and then transpose to (B,H,L,HD)
        q=q.view(b_size,seq_l,self.n_heads,self.n_head_dim).transpose(1,2)
        k=k.view(b_size,seq_l,self.n_heads,self.n_head_dim).transpose(1,2)
        v=v.view(b_size,seq_l,self.n_heads,self.n_head_dim).transpose(1,2)
        
        #
        if torch_attention:#build in attention
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        else: #Manual implementation
            #compute scores as q*kt. q shape is (B,H,L,HD) and  kt shape is (B,H,HD,L)
            #the result is the shape (B,H,L,L)
            scores=torch.matmul(q,k.transpose(-2,-1))/self.scale
            #mask
            if mask is not None:
                scores=scores.masked_fill(mask==0,float('-inf'))

            att_w=torch.softmax(scores,dim=-1)
            #att_w shape is (B,H,L,L). y's shape is (B,H,L,HD). Result is (B,H,L,HD)
            y=torch.matmul(att_w,v)
            
        #reshape to (B,L,H,HD) and then (B,L,D)
        y=y.transpose(1,2).contiguous().view(b_size,seq_l,n_embd)
        out=self.c_proj(y)
                
        return out

class CausalSelfAttn(nn.Module):
    
    def __init__(self,config:GPTConfig,flash=True):
            
        super().__init__()
        
        self.config=config
        
        #dimensionality of each head
        self.n_head_dim=config.n_embd // config.n_head
        
        #Linear
        self.c_attn=nn.Linear(config.n_embd,3*config.n_embd,bias=config.bias)
        
        #final layer
        self.c_proj=nn.Linear(config.n_embd,config.n_embd)
        
        #dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        if flash & hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            self.flash=True
        else:
            
            self.flash=False
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # adding 2 exta dimensions to match the att shape (B,HD,L,L)
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        
    def forward(self,x, mask=None):
        
        b_size, seq_l, n_embd = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        
        #linear proj
        q,k,v=self.c_attn(x).split(self.config.n_embd,dim=2)
        
        #reshape (B,L,D) --> (B,L,H,HD) and then transpose to (B,H,L,HD)
        q=q.view(b_size,seq_l,self.config.n_head,self.n_head_dim).transpose(1,2)
        k=k.view(b_size,seq_l,self.config.n_head,self.n_head_dim).transpose(1,2)
        v=v.view(b_size,seq_l,self.config.n_head,self.n_head_dim).transpose(1,2)
        
        if self.flash:#built in attention
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,dropout_p=self.config.dropout if self.training else 0, is_causal=True)
        else: #Manual implementation
            
            #scale to stabilize gradient
            scale=math.sqrt(self.n_head_dim)
            
            #compute scores as q*kt. q shape is (B,H,L,HD) and  kt shape is (B,H,HD,L)
            #att: (B,H,L,L)
            att=torch.matmul(q,k.transpose(-2,-1))/scale
            
            #Get bias submatrix with the shape (1,1,L,L)
            #Replace uper (right) triangular side from 0 to -inf (-inf is later converted to 0 by softmax)
            att = att.masked_fill(self.bias[:,:,:seq_l,:seq_l] == 0, float('-inf'))

            att=torch.softmax(att,dim=-1)
            
            att = self.attn_dropout(att)
            
            #att_w: (B,H,L,L). y: (B,H,L,HD). Result: (B,H,L,HD)
            y=torch.matmul(att,v)
            
        #Reshape to (B,L,H,HD) and then (B,L,D)
        y=y.transpose(1,2).contiguous().view(b_size,seq_l,n_embd)
        
        y = self.resid_dropout(self.c_proj(y))
        
        return y

class MLP(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        #based on Transformer paper, MLP inner dimm is 4x the model dimm
        self.c_fc = nn.Linear(config.n_embd,4*config.n_embd,bias=config.bias)
        self.gelu=nn.GELU()
        self.c_proj=nn.Linear(4*config.n_embd,config.n_embd)
        self.dropout=nn.Dropout(config.dropout)
    def forward(self,x):
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)
        x=self.dropout(x)
        return x
    
class LayerNorm(nn.Module):
    def __init__(self,ndim,bias):
        #layer norm with optional bias
        super().__init__()
        self.weight=nn.Parameter(torch.ones(ndim))
        self.bias=nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self,x):
        return nn.functional.layer_norm(x,self.weight.shape,self.weight,self.bias,1e-5)

class TransformerBlock(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        self.ln_1=LayerNorm(config.n_embd,config.bias)
        self.attn=CausalSelfAttn(config)
        self.ln_2=LayerNorm(config.n_embd,config.bias)
        self.mlp=MLP(config)
        
    def forward(self,x):
        x=x+self.attn(self.ln_1(x))
        x=x+self.mlp(self.ln_2(x))
        return x
    
from transformers import GPT2LMHeadModel
class HF_GPT2(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        self.config=config
        self.transformer=nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size,config.n_embd),
            wpe=nn.Embedding(config.block_size,config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer)),
            ln_f=LayerNorm(config.n_embd,bias=config.bias)
        ))
        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)
        
        self.transformer.wte.weight = self.lm_head.weight
        
        #PyTorch way to initialize the weights of each layer(module). 
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = HF_GPT2(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask. buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        
        # basically the openai checkpoints use a "Conv1D" module with kernel size=1, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model
    
    def forward(self,idx,targets=None):
        #batch size and sequence length
        b,t=idx.size()
        #Text and position embeeding.
        pos=torch.arange(0,t,device=idx.device)
        tok_embd=self.transformer.wte(idx)
        pos_embd=self.transformer.wpe(pos)
        y=self.transformer.drop(tok_embd+pos_embd)

        #transformer blocks
        for block in self.transformer.h:
            y=block(y)
        
        #norm
        y=self.transformer.ln_f(y)

        #final linear
        y=self.lm_head(y)
        
        loss=None
        if targets is not None:
            #Note: at initialization, cross_entropy loss=-ln(1/vocab_size)=-ln(1/50257)=10.82
            loss=nn.functional.cross_entropy(y.view(-1,y.size(-1)),targets.view(-1), ignore_index=-1)
        
        return y, loss
    
    def generate(self,idx,max_length,temp=1):
        for _ in range (max_length):
            #return (B,L,Voc_size)
            logits,_=self(idx)
            
            #Transformers compute logits for all tokens in the input secuence.
            #Only the last logit is relevant, logits for the earlies tokens are predictions for tokens 
            #already generated.  
            logits=logits[:,-1,:]/temp
            probs=nn.functional.softmax(logits,dim=-1)
            #sample
            idx_next=torch.multinomial(probs,num_samples=1)
            #append sample to the sequence and repeat
            idx=torch.cat((idx,idx_next),dim=1)    
        return idx

import tiktoken
class DataLoader():
    def __init__(self,B,T):
        self.B=B
        self.T=T
        pass
    def load_txt(self,fpath):
        with open(fpath,'r') as f:
            text=f.read()        
        enc=tiktoken.get_encoding('gpt2')
        tokens=enc.encode(text)
        self.tokens=torch.tensor(tokens)
        self.curr_pos=0
    def next_batch(self):
        buf=self.tokens[self.curr_pos:self.curr_pos+self.B*self.T+1]
        x=(buf[:-1]).view(self.B,self.T)
        y=(buf[1:]).view(self.B,self.T)
        self.curr_pos=self.curr_pos+self.B*self.T
        if self.curr_pos+(self.B*self.T+1)>len(self.tokens):
            self.curr_pos=0
        return x,y


