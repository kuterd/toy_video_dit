from tqdm import tqdm
from torch import nn, optim
import torch
import torchvision

import utils
import model

import numpy as np

torch.set_float32_matmul_precision('high')

width = 64
height = 64
frames = 20

patch_w = 4
patch_h = 4

dit_model = model.ToyVideoDiT(patch_w, patch_h, depth=16, hidden_multiplier=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dit_model = dit_model.to(device)
dit_model = torch.compile(dit_model)

dataset = torchvision.datasets.MovingMNIST(
    root="moving_mnist/", download=True
)

batch_size = 16


dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True
)

optimizer = optim.Adam(dit_model.parameters(), lr=0.001)
criterion = nn.MSELoss()

schedule = utils.ScheduleLinear(1000)

num_epochs = 80
checkpoint_interval = 1

x_ids = utils.generate_ids(batch_size, frames, patch_w, patch_h, width, height, device)

for epoch in tqdm(range(num_epochs)):
    losses = []
    for x in tqdm(dataloader):
        x = x.to(device).to(torch.float32) / 255.0
        sigma = schedule.sample_batch(x)
        
        corrupted_x, noise = utils.corrupt(x, sigma)

        corrupted_x = utils.patchify(corrupted_x, patch_w, patch_h)
        target = utils.patchify(x - noise, patch_w, patch_h)
        
        outputs = dit_model(corrupted_x, x_ids, sigma)
        loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    avg_loss = np.array(losses).mean()
    print(f'Epoch [{epoch+1}/{num_epochs}], AVG Loss: {avg_loss}')

    if (epoch + 1) % checkpoint_interval == 0:
        checkpoint_filename = f"moving_mnist_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': dit_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_filename)
        print(f"Checkpoint saved to {checkpoint_filename}")
