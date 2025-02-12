# simple_gpt2

So this is my 1st attempt at LLMs, I followed Karpathy great tutorial here https://www.youtube.com/watch?v=l8pRSuU81PU&t=8371s
Some portions of the code are copied, and others are rewritten in my style. This version is simpler (and less robust) than the version from Karpathy.

Response from my model to the prompt "I'm an artificial intelligence":
![a](images/my_response.png)

Response to Huggingface gpt 2 model:
![nanoGPT](images/gpt2_resp.png)

Losses
![nanoGPT](images/loss.png)

## Dependencies 

## Training
Trained on ~1.6B tokens over Oxford English Dictionary,fineweb and  dailydialog.
~2k steps of 0.5M tokens batch size.
Training duration ~2days

## closing notes
This is just the starting point to take us off the ground.
