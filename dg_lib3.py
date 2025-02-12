import torch.nn as nn
import torch
import math
import tiktoken

from dataclasses import dataclass
from safetensors.torch import save_file,load_file
import time

@dataclass
class GPTConfig:
    #vocab_size:int=50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    vocab_size:int=50257
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
        
    def forward(self,x):
        
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
        
        #per Transformer paper
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

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
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
    
    def get_device(self):
        return next(self.parameters()).device
    
    def generate(self,prompt,max_length=30,num_return_sequences=2,temp=1):

        #encode
        enc=tiktoken.get_encoding('gpt2')
        tokens=enc.encode(prompt)
        
        #list to tensor
        tokens=torch.tensor(tokens,dtype=torch.long)
        tokens=tokens.unsqueeze(0).repeat(num_return_sequences,1)

        device=next(self.parameters()).device
        idx=tokens.to(device)

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

            # Check which sequences have reached the EOT token
            eot_mask = (idx_next == enc.eot_token).squeeze(1)  # Shape: (batch_size,)

            # If all sequences in the batch have reached EOT, stop early
            if eot_mask.all():
                break

            #append sample to the sequence and repeat
            idx=torch.cat((idx,idx_next),dim=1)    
        
        #decode
        response=[] 
        for i in range(num_return_sequences):
            tokens=idx[i,:max_length].tolist()
            decoded=enc.decode(tokens)
            response.append(decoded)

        return response
    
    def save_safetensor(self,path):
        # temporary clone before saving 
        self.transformer.wte.weight=torch.nn.Parameter(self.transformer.wte.weight.clone())
        save_file(self.state_dict(), path)
        #revert back the shared memory
        self.transformer.wte.weight = self.lm_head.weight


    
    def load_safetensor(self,path):
        state_dict=load_file(path)
        # Re-assign nn.Parameter to the transformer weights 
        state_dict['transformer.wte.weight'] = torch.nn.Parameter(state_dict['transformer.wte.weight'])
        missing_keys, unexpected_keys=self.load_state_dict(state_dict)
        #shared memory
        self.transformer.wte.weight = self.lm_head.weight




import pickle
import gzip
from datasets import load_dataset
class FineWebDataset():
    def __init__(self):
        self.records=[]
    def download_records(self,samples_per_file=10000,file_count=4,folder='../../data/'):
        fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        
        sample = []
        rec_id=0
        file_id=0
        for i, record in enumerate(fw):
            sample.append(record)  # Collect the record
            rec_id+=1
            print(f"file: {file_id},  rec: {i}",end='\r')
            if rec_id >= samples_per_file:  # Stop after 5 records
                # Save records to a compressed file
                with gzip.open(f"{folder}fineweb_records_{samples_per_file}_{file_id}.pkl.gz", "wb") as f:
                    pickle.dump(sample, f)
                sample=[]
                rec_id=0
                file_id+=1
                if file_id>=file_count:
                    break
    
    def load_local(self,fpath):
        # Load records from the compressed file
        with gzip.open(fpath, "rb") as f:
            self.records = pickle.load(f)
        return self.records


            
from torch.utils.data import Dataset, DataLoader, random_split
import os.path, json

import concurrent.futures
class DataLoader():
    def __init__(self,B,T):
        self.records=[]
        self.B=B
        self.T=T
        self.tokens_per_batch=B*T
        self.batches_per_epoch=0
        self.training=[]
        self.validate=[]    
        self.train_ratio=0.8

        self.curr_pos=0
        self.curr_train_pos=0
        self.curr_val_pos=0

    def print_train_summary(self):
        self.batches_per_epoch=len(self.tokens)//self.tokens_per_batch

        print(f"tokens: {len(self.tokens)}")
        print(f"batch size: {self.tokens_per_batch}")
        print(f"Batches: {self.batches_per_epoch}")        

    def load_txt(self,fpath,train_ratio=0.9):
        with open(fpath,'r', encoding="utf8") as f:
            text=f.read()    
            records=text.split("\n")
            self.records+=records    
        

    def load_local_fineweb(self,fpath,train_ratio=0.9,num_threads=24):    
        print("loading data ...")
        dataset = FineWebDataset()
        records = dataset.load_local(fpath)
        records=[rec["text"] for rec in records]
        self.records+=records
        print("data loaded")

    def load_dailydialog(self,fpath):        
        with open(fpath,'r', encoding="utf8") as f:
            text=f.read()        
        dialogs=text.split('\n')
        for i in range(len(dialogs)):
            dialogs[i]=dialogs[i].split(" __eou__ ")
            perid=0
            for j in range(len(dialogs[i])):
                dialogs[i][j]=f"person{perid}: {dialogs[i][j]}"
                if perid==0:
                    perid=1
                else:
                    perid=0
        records=[]        
        for dialog in dialogs:
            record=""         
            for line in dialog:
                record+=line+"\n"
            records.append(record)
        self.records+=records
        #print("a")       

    def rand_split(self,train_ratio=0.8):
        l=len(self.records)
        train_size = int(l * train_ratio)
        val_size = l - train_size
        train_indices, val_indices = random_split(range(l), [train_size, val_size])        
        self.train_records=[self.records[i] for i in train_indices]
        self.val_records=[self.records[i] for i in val_indices]

    def __encode_records(self,records,num_threads=24):        
        # Function to encode a chunk of records
        def encode_chunk(chunk_index, records_chunk):
            chunk_tokens = []
            for i,rec in enumerate(records_chunk):
                t = enc.encode(rec,allowed_special={'<|endoftext|>'})
                t.append(enc.eot_token)
                chunk_tokens += t
                if chunk_index==0:               
                    print(f"thread progress: {i}/{len(records_chunk)}", end='\r')
            
            if chunk_index==0:
                print(f"\nTo tensor chunck")
            return chunk_index, torch.tensor(chunk_tokens)            

        enc = tiktoken.get_encoding('gpt2')
        
        
        # Divide the records into chunks for each thread
        chunk_size = len(records) // num_threads
        chunks = [records[i:i + chunk_size] for i in range(0, len(records), chunk_size)]
        
        print("encoding ...")
        # Use ThreadPoolExecutor to execute each chunk in a separate thread
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Execute encode_chunk for each chunk in parallel
            #results = list(executor.map(encode_chunk, chunks))
            results = list(executor.map(lambda x: encode_chunk(x[0], x[1]), enumerate(chunks)))
        print("Assembling tokens...")
        results.sort(key=lambda x: x[0])  # Sort by original chunk index        
        tokens = torch.cat([chunk[1] for chunk in results])
        return tokens
    
    def encode(self):
        self.train_tokens=self.__encode_records(self.train_records)
        self.val_tokens=self.__encode_records(self.val_records)


    def next_batch(self):
        buf=self.tokens[self.curr_pos:self.curr_pos+self.B*self.T+1]
        x=(buf[:-1]).view(self.B,self.T)
        y=(buf[1:]).view(self.B,self.T)
        self.curr_pos=self.curr_pos+self.B*self.T
        if self.curr_pos+(self.B*self.T+1)>len(self.tokens):
            self.curr_pos=0
        return x,y
    
    def next_train_batch(self):
        buf=self.train_tokens[self.curr_train_pos:self.curr_train_pos+self.B*self.T+1]
        x=(buf[:-1]).view(self.B,self.T)
        y=(buf[1:]).view(self.B,self.T)
        self.curr_train_pos=self.curr_train_pos+self.B*self.T
        if self.curr_train_pos+(self.B*self.T+1)>len(self.train_tokens):
            self.curr_train_pos=0
        return x,y
    
    def next_val_batch(self):
        buf=self.val_tokens[self.curr_val_pos:self.curr_val_pos+self.B*self.T+1]
        x=(buf[:-1]).view(self.B,self.T)
        y=(buf[1:]).view(self.B,self.T)
        self.curr_val_pos=self.curr_val_pos+self.B*self.T
        if self.curr_val_pos+(self.B*self.T+1)>len(self.val_tokens):
            self.curr_val_pos=0
        return x,y

    def get_train_token_lenght(self):
        return len(self.train_tokens)

class Trainer_base():
#Basic training for reference
    def __init__(self):
        pass

    def train(self,model,data_loader,steps=50, lr=3e-4):

        optimizer=torch.optim.AdamW(model.parameters(),lr=lr,betas=(0.9,0.95),eps=1e-8)
        t0=time.time()

        for it in range(steps):
            optimizer.zero_grad()
            
            x,y=data_loader.next_batch()
            x=x.to('cuda')
            y=y.to('cuda')   
        
            logits,loss=model(x,y)
            loss.backward()
            
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            torch.cuda.synchronize()
            t1=time.time()
            dt=t1-t0
            t0=t1
            tokens_processed=data_loader.B*data_loader.T
            tokens_per_sec=tokens_processed/dt
            print(f"it: {it}, loss: {loss:.1f},lr={lr:.4e},dt={dt*1000:.1f}ms, norm:{norm:.1f}, tok/sec={tokens_per_sec:.1f}")

class LR_cosine_with_warmup():
    def __init__(self,max_lr=3e-4,min_lr=6e-5,warmup_iters=100,lr_decay_iters=200):
        self.warmup_iters=warmup_iters
        self.max_lr=max_lr
        self.min_lr=min_lr
        self.lr_decay_iters=lr_decay_iters
        
    def get_lr(self,it):
        if it<self.warmup_iters:
            return self.max_lr*(it+1)/(self.warmup_iters+1)
        if it>self.lr_decay_iters:
            return self.min_lr
        decay_ratio=(it-self.warmup_iters)/(self.lr_decay_iters-self.warmup_iters)
        assert 0<=decay_ratio<=1
        coeff=0.5*(1+math.cos(math.pi*decay_ratio))
        return self.min_lr+coeff*(self.max_lr-self.min_lr)


import inspect
import matplotlib.pyplot as plt
class Trainer():
    def __init__(self):
        self.lr_scheduler=LR_cosine_with_warmup(warmup_iters=10,lr_decay_iters=40)
        self.batch_size=2**19
        self.checkpoint={}

    def configure_optimizers(self,model, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in model.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def train(self,model,data_loader,steps=50):

        optimizer=self.configure_optimizers(model,0.1,self.lr_scheduler.max_lr,(0.9,0.95),str(model.get_device()))
        step=0
        train_losses=[]
        val_losses=[]
        #check if we have configured the optimizer before
        if self.checkpoint:
            optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            step=self.checkpoint['step']
            model.load_state_dict(self.checkpoint['model_state_dict'])
            model.transformer.wte.weight = model.lm_head.weight

            train_losses=self.checkpoint['train_losses']
            val_losses=self.checkpoint['val_losses']
        
        lr_scheduler=self.lr_scheduler
        t0=time.time()

        B=data_loader.B
        T=data_loader.T
        batch_size=self.batch_size

        grad_accum_steps=batch_size//(B*T)
        
        for it in range(step,step+steps):
            
            model.train()
            optimizer.zero_grad()
            loss_accum=0
            for mini_batch in range(grad_accum_steps):
                x,y=data_loader.next_train_batch()
                x=x.to('cuda')
                y=y.to('cuda')   
            
                #with torch.autocast(device_type=str(model.get_device()),dtype=torch.bfloat16):
                logits,loss=model(x,y)
                loss=loss/ grad_accum_steps#normalize
                loss_accum+=loss.detach() 
                loss.backward()
                print (f"it: {it}, mini batch: {mini_batch+1}/{grad_accum_steps}",end='\r')
            
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            lr=lr_scheduler.get_lr(it)
            for param_group in optimizer.param_groups:
                param_group['lr']=lr
            optimizer.step()
            torch.cuda.synchronize()
            train_losses.append(loss_accum.item())
            t1=time.time()
            dt=t1-t0
            t0=t1
            tokens_processed=B*T*grad_accum_steps
            tokens_per_sec=tokens_processed/dt
            

            #validations
            model.eval()
            
            val_secs=3
            with torch.no_grad():
                loss_accum_val=0
                for mini_batch in range(val_secs):
                    x,y=data_loader.next_val_batch()
                    x=x.to('cuda')
                    y=y.to('cuda')   
            
                    #with torch.autocast(device_type=str(model.get_device()),dtype=torch.bfloat16):
                    logits,loss=model(x,y)
                    loss=loss/ val_secs#normalize
                    loss_accum_val+=loss.detach()
                val_losses.append(loss_accum_val.item()) 
            
            print(f"it: {it}, t_loss: {loss_accum:.1f},v_loss: {loss_accum_val:.1f},lr={lr:.4e},dt={dt*1000:.1f}ms, norm:{norm:.1f}, tok/sec={tokens_per_sec:.1f}")

        # Save checkpoint
        self.checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': it + 1,
            'train_losses':train_losses,
            'val_losses':val_losses
        }
        
        plt.plot(range(1, step+steps+1), train_losses, label='Training Loss')
        plt.plot(range(1, step+steps+1), val_losses, label='Validation Loss')
        plt.xlabel('steps')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid()
        plt.show()
    def save_checkpoint(self,checkpoint_path  ):
        torch.save(self.checkpoint, checkpoint_path)

    def load_checkpoint(self,checkpoint_path):
        self.checkpoint = torch.load(checkpoint_path)
        


#--------------------------
class VidPatchEmbd(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, embed_dim=768,num_frames=10):
        super().__init__()
        self.patch_size=patch_size
        self.num_patches=int((img_size/patch_size)**2)
        self.patch_embd=nn.Conv2d(in_channels=in_channels,out_channels=embed_dim,kernel_size=patch_size,stride=patch_size)

        self.pos_embd=nn.Parameter(torch.rand(1,self.num_patches,embed_dim))

        # Learnable temporal positional embedding
        self.temp_pos_embd = nn.Parameter(torch.randn(1, num_frames, 1, embed_dim))
    
    def forward(self,x):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # Flatten temporal dimension to process frames independently. Computation trick to proccess each frame inependently
        #assuming a larger batch size
        x = x.view(B * T, C, H, W)  # Shape: (B*T, C, H, W)


        x=self.patch_embd(x) # shape (B,embd_dimm,H/patch, W/patch)        
        #flat last 2 dims(H,W) into total numpatches and then transpose 1 and 2. 
        x=x.flatten(2).transpose(1,2) #shape (B,num_patches,endb_dimm)
        
        x=x+self.pos_embd #shape (B,L,endb_dimm)

        # Reshape back to include temporal dimension
        x = x.view(B, T, self.num_patches, -1)  # Shape: (B, T, num_patches, embed_dim)

        x=x+self.temp_pos_embd # Shape (B,T,num_patches,embd_dim)
        
        return x

class VidMLP(nn.Module):
    def __init__(self, n_embd, bias=True, dropout=0.1):
        super().__init__()
        # Based on the Transformer paper, MLP inner dimension is 4x the model dimension
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (B, T, num_patches, n_embd)
        B, T, num_patches, n_embd = x.shape

        # Flatten the temporal and spatial dimensions
        x = x.contiguous().view(B, T * num_patches, n_embd)  # Shape: (B, T * num_patches, n_embd)

        # Apply MLP layers
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)

        # Reshape back to original shape
        x = x.view(B, T, num_patches, n_embd)  # Shape: (B, T, num_patches, n_embd)

        return x
    
class VidFramePrediction(nn.Module):
    def __init__(self, img_size=256, patch_size=16, embed_dim=768, out_channels=3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels

        # Linear layer to project embedding dimension back to the output channel dimension
        self.linear_proj = nn.Linear(embed_dim, patch_size * patch_size * out_channels)

        # Upsample patches back to original image dimensions
        self.unpatchify = nn.PixelShuffle(patch_size)

    def forward(self, x):
        # x shape: (B, T, num_patches, n_embd)
        B, T, num_patches, n_embd = x.shape

        # Ensure embedding dimension matches
        assert n_embd == self.embed_dim, "Embedding dimension mismatch."
        
        # Optionally: Combine temporal information if T > 1 (if the decoder is for video)
        # You can average or apply some operation over T dimension (e.g., max-pooling)
        # In this case, we'll assume it's per-frame, but you may change depending on the task
        if T > 1:
            x = x.mean(dim=1)  # Aggregate information across the T dimension (optional)

        # Convert spatial patches back to image patches
        x = self.linear_proj(x)  # Shape: (B, num_patches, patch_size * patch_size * out_channels)

        # Reshape to match spatial patches
        H_p = W_p = int(self.img_size // self.patch_size)
        x = x.view(B, H_p, W_p, self.patch_size, self.patch_size, self.out_channels)  # (B, H_p, W_p, patch_h, patch_w, C)

        # Rearrange to merge spatial patches into full image
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, H_p, patch_h, W_p, patch_w)
        x = x.contiguous().view(B, self.out_channels, self.img_size, self.img_size)  # (B, C, H, W)

        return x

class VidDecoderWithAttention(nn.Module):
    def __init__(self, img_size=256, patch_size=16, embed_dim=768, out_channels=3, num_heads=8):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels

        # Linear layer to project embedding dimension back to the output channel dimension
        self.linear_proj = nn.Linear(embed_dim, patch_size * patch_size * out_channels)

        # Multihead Attention for temporal aggregation (across frames)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        # Upsample patches back to original image dimensions
        self.unpatchify = nn.PixelShuffle(patch_size)

    def forward(self, x):
        # x shape: (B, T, num_patches, n_embd)
        B, T, num_patches, n_embd = x.shape

        # Ensure embedding dimension matches
        assert n_embd == self.embed_dim, "Embedding dimension mismatch."
        
        # Optionally: Flatten num_patches into a single dimension to apply attention across
        x = x.view(B, T * num_patches, n_embd)  # Shape: (B, T * num_patches, n_embd)

        # Apply temporal attention across frames
        # Query, Key, and Value all come from the same input (x)
        x,_= self.attention(x, x, x)  # (B, T * num_patches, n_embd)

        # Optionally, reshape back to (B, num_patches, n_embd) if necessary
        x = x.view(B, num_patches, T, n_embd)  # Shape: (B, num_patches, T, n_embd)
        #x = x.view(B, T,num_patches,  n_embd)  # Shape: (B, num_patches, T, n_embd)

        # Convert spatial patches back to image patches
        x = self.linear_proj(x)  # Shape: (B, num_patches, patch_size * patch_size * out_channels)

        # Reshape to match spatial patches
        H_p = W_p = int(self.img_size // self.patch_size)
        x=x.view(B, T, H_p, W_p, self.patch_size, self.patch_size, self.out_channels)  # (B, T, H_p, W_p, patch_h, patch_w, C)        

        # Rearrange to merge spatial patches into full image
        x = x.permute(0, 1, 6, 2, 4, 3, 5)  # (B, T, C, H_p, patch_h, W_p, patch_w)
        x = x.contiguous().view(B,T, self.out_channels, self.img_size, self.img_size)  # (B, C, H, W)

        return x

class CausalSelfAttn_2(nn.Module):
    
    def __init__(self,n_embd,n_head,bias,dropout,block_size,flash=True):
            
        super().__init__()
        
        self.n_embd=n_embd 
        self.n_head=n_head
        self.bias=bias
        self.dropout=dropout
        self.block_size=block_size
        
        #dimensionality of each head
        self.n_head_dim=n_embd // n_head
        
        #Linear
        self.c_attn=nn.Linear(n_embd,3*n_embd,bias=bias)
        
        #final layer
        self.c_proj=nn.Linear(n_embd,n_embd)
        
        #dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        if flash & hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            self.flash=True
        else:
            self.flash=False
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # adding 2 exta dimensions to match the att shape (B,HD,L,L)
            self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))
        
    def forward(self,x):
        
        b_size, seq_l, n_embd = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        
        #linear proj
        q,k,v=self.c_attn(x).split(self.n_embd,dim=2)
        
        #reshape (B,L,D) --> (B,L,H,HD) and then transpose to (B,H,L,HD)
        q=q.view(b_size,seq_l,self.n_head,self.n_head_dim).transpose(1,2)
        k=k.view(b_size,seq_l,self.n_head,self.n_head_dim).transpose(1,2)
        v=v.view(b_size,seq_l,self.n_head,self.n_head_dim).transpose(1,2)
        
        if self.flash:#built in attention            
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,dropout_p=self.dropout if self.training else 0, is_causal=True)
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

class VidPred(nn.Module):
    def __init__(self):
        super().__init__()

        img_h,img_w=256,256
        ch=3
        frames=10                
        self.end=VidPatchEmbd(256,16,3,768,frames)        
        self.mlp=VidMLP(768)        
        self.decoder = VidDecoderWithAttention(img_size=img_h, patch_size=16, embed_dim=768, out_channels=ch)        

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def forward(self,x,targets=None):
        #x and target shapes are (B,T,C,H,W)
        x=self.end(x)
        x=self.mlp(x)
        y= self.decoder(x) #shape(B,T,C,H,W)
        
        loss=None
        if targets is not None:
            # Compute MSE loss between predicted and target frames
            # No need to reshape since both y and targets are (B,T,C,H,W)
            loss = nn.functional.mse_loss(y, targets)
            
            # Optionally, you could add L1 loss for sharper predictions
            # alpha = 0.5  # Weight between MSE and L1
            # l1_loss = nn.functional.l1_loss(y, targets)
            # loss = alpha * loss + (1 - alpha) * l1_loss
            
            # You might also want to add a perceptual loss using a pretrained network
            # This helps generate more visually pleasing results
        
        return y.detach(), loss
    
    def get_device(self):
        return next(self.parameters()).device
    
    def generate(self,prompt,max_length=30,num_return_sequences=2,temp=1):
        pass

        
    
    def save_safetensor(self,path):
        # temporary clone before saving 
        self.transformer.wte.weight=torch.nn.Parameter(self.transformer.wte.weight.clone())
        save_file(self.state_dict(), path)
        

    
    def load_safetensor(self,path):
        state_dict=load_file(path)
        # Re-assign nn.Parameter to the transformer weights 
        state_dict['transformer.wte.weight'] = torch.nn.Parameter(state_dict['transformer.wte.weight'])
        missing_keys, unexpected_keys=self.load_state_dict(state_dict)
        


import cv2
import time
import torch
import numpy as np
class dg_vision():
    def __init__(self):
        pass
    @staticmethod
    def video_to_tensor(file, num_frames=-1):
        # Open the video file
        cap = cv2.VideoCapture(file)

        if not cap.isOpened():
            print("Error: Could not open video file.")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if num_frames==-1:
            num_frames=total_frames


        frames = []
        frame_count = 0


        while frame_count < num_frames:
            ret, frame = cap.read()

            if not ret:
                print("Error: Couldn't read a frame.")
                break
            
            # Convert frame from BGR (OpenCV format) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_rgb = frame_rgb / 255.0
            # Convert frame to tensor and append to the list
            frame_tensor = torch.tensor(frame_rgb,dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            frames.append(frame_tensor)

            frame_count += 1

        cap.release()

        # Stack frames into one tensor (B=1, T=num_frames, C, H, W)
        video_tensor = torch.cat(frames, dim=0)  # Shape: (T, C, H, W)
        video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension: (B, T, C, H, W)

        return video_tensor,fps,total_frames

    @staticmethod
    def resize_and_crop(video_tensor, target_size=(256, 256)):
        # Get the video shape: B, T, C, H, W
        B, T, C, H, W = video_tensor.shape

        resized_frames = []

        for t in range(T):
            frame = video_tensor[0, t].permute(1, 2, 0).numpy()  # Convert to (H, W, C)

            # Calculate scale for resizing
            scale = max(target_size[0] / frame.shape[0], target_size[1] / frame.shape[1])

            # Resize the frame keeping the aspect ratio
            new_w = int(frame.shape[1] * scale)
            new_h = int(frame.shape[0] * scale)
            resized_frame = cv2.resize(frame, (new_w, new_h))

            # Crop the center to match the target size
            top = (new_h - target_size[0]) // 2
            bottom = new_h - target_size[0] - top
            left = (new_w - target_size[1]) // 2
            right = new_w - target_size[1] - left

            cropped_frame = resized_frame[top:top+target_size[0], left:left+target_size[1]]

            # Convert the cropped frame back to a tensor
            cropped_frame_tensor = torch.tensor(cropped_frame).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            resized_frames.append(cropped_frame_tensor)

        # Stack frames into one tensor (B=1, T=num_frames, C, H, W)
        resized_video_tensor = torch.cat(resized_frames, dim=0)  # Shape: (T, C, H, W)
        resized_video_tensor = resized_video_tensor.unsqueeze(0)  # Add batch dimension: (B, T, C, H, W)

        return resized_video_tensor

    # Assume img is a tensor of shape (T, C, H, W)
    @staticmethod
    def display_video(video_tensor, fps=30):
        T, C, H, W = video_tensor.shape  # Extract dimensions

        for t in range(T):  # Iterate through frames
            img_np = video_tensor[t].permute(1, 2, 0).numpy()  # Convert to (H, W, C)

            img_np = (img_np * 255).astype(np.uint8)  # Convert to uint8 (if normalized)

            cv2.imshow('Video', img_np)  # Show frame
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):  # Delay based on FPS
                break  # Press 'q' to exit

        cv2.destroyAllWindows()  # Cleanup
