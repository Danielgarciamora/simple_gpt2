import torch
# lets do som predictions
import tiktoken


from dg_lib3 import DataLoader, Trainer,Trainer_base,VidPatchEmbd,VidMLP,VidDecoderWithAttention,VidPred

import matplotlib.pyplot as plt
from dg_lib3 import dg_vision

img_h,img_w=256,256
ch=3
frames=10


img=torch.rand(1,frames,ch,img_h,img_w)

vid,fps,frame_count=dg_vision.video_to_tensor("../../data/video/video_04064.mp4",frames)
print(frame_count)
img = dg_vision.resize_and_crop(vid, target_size=(256, 256))
target=img



model=VidPred()
out,loss=model(img,target)


#torch.compile()

import torch.optim as optim



optimizer = optim.Adam(model.parameters(), lr=0.001)
# Training loop
for epoch in range(1000):
    model.train()  # Ensure the model is in training mode

    # Zero the gradients before the backward pass
    optimizer.zero_grad()

    # Forward pass
    out, loss = model(img, target)

    # Compute the loss (if it's not computed already by the model)
    # In case model returns raw output, compute loss manually
    # loss = criterion(out, target)

    # Backward pass to compute gradients
    loss.backward()

    # Update the weights with the optimizer
    optimizer.step()

    # Print the loss at every epoch
    print(f"Epoch [{epoch+1}/{10}], Loss: {loss.item()}")

dg_vision.display_video(img.squeeze(0),25)
dg_vision.display_video(out.squeeze(0),25)





# Convert tensor for plotting
#img_np = img.squeeze(0).permute(1, 2, 0).numpy()

# Create the plot
#plt.figure(figsize=(8, 8))
#plt.imshow(img_np)
#plt.axis('off')
#plt.title('img')
#plt.show()


from dg_lib3 import dg_vision



vid,fps,frame_count=dg_vision.video_to_tensor("../../data/video/video_04064.mp4")
print(frame_count)
vid2 = dg_vision.resize_and_crop(vid, target_size=(256, 256))
dg_vision.display_video(vid2.squeeze(0),25)
