import numpy as np
import torch
w1=np.array([[0.2,-0.3,0.1],[.4,.2,-.1],[-.2,.3,.5]])
x=np.array([1,.5,-1])
b=np.array([0.1,0.2,-0.1])

#calculate z1, 
#col array can be created using reshape(-1,1)
z1=torch.tensor(w1,dtype=torch.float32) @ torch.tensor(x,dtype=torch.float32).reshape(-1,1)
z1=z1+torch.tensor(b,dtype=torch.float32).reshape(-1,1)
s1=torch.nn.functional.relu(z1)
print(s1)


w2=np.array([[-.4,0.2,0.3],[.1,-.2,0.4],[-.3,.1,-.5]])
b2=np.array([0.1,-0.1,0.2])

z2=torch.tensor(w2,dtype=torch.float32) @ torch.tensor(s1,dtype=torch.float32).reshape(-1,1)
z2=z2+torch.tensor(b2,dtype=torch.float32).reshape(-1,1)
s2=torch.nn.functional.relu(z2)


w3=np.array([[.3,-0.2,0.4],[-.1,.5,-0.3],[.2,-.4,.1]])
b3=np.array([-0.2,0.3,0.1])

z3=torch.tensor(w3,dtype=torch.float32) @ torch.tensor(s2,dtype=torch.float32).reshape(-1,1)
z3=z3+torch.tensor(b3,dtype=torch.float32).reshape(-1,1)
s3=torch.nn.functional.softmax(z3,dim=0)

y=np.array([0,1,0])
y=torch.tensor(y,dtype=torch.float32).reshape(-1,1)

l=torch.nn.functional.mse_loss(s3,y)


#derivative
#Calculate the Jacobian of the softmax in a single line 


jacobian_m=np.zeros((len(s3),len(s3)))
for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = s3[i] * (1-s3[i])
            else: 
                jacobian_m[i][j] = -s3[i]*s3[j]

jacobian_m =torch.tensor(jacobian_m,dtype=torch.float32)
print(jacobian_m)
print(s3)


dl=s3-y

d3=jacobian_m @ dl



###
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

