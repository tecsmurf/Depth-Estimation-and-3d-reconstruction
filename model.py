import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import DPTForDepthEstimation, DPTImageProcessor
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class DepthDataset(Dataset):
    def __init__(self, image_dir, depth_dir, processor):
        self.image_paths = sorted(os.listdir(image_dir))
        self.depth_paths = sorted(os.listdir(depth_dir))
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, self.image_paths[idx])).convert("RGB")
        depth_path = os.path.join(self.depth_dir, self.depth_paths[idx])

        
        if depth_path.endswith(".npy"):
            depth_map = np.load(depth_path)
        else:  
            depth_map = np.array(Image.open(depth_path).convert("L"), dtype=np.float32)

        inputs = self.processor(images=image, return_tensors="pt")
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0), 
            "depth_map": torch.tensor(depth_map, dtype=torch.float32).unsqueeze(0),
        }


image_dir = "dataset/images"
depth_dir = "dataset/depths"

device = torch.device("cuda")

processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(device)


for param in model.dpt.parameters():
    param.requires_grad = False


dataset = DepthDataset(image_dir, depth_dir, processor)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.head.parameters(), lr=1e-4)


num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    print(f"Starting Epoch {epoch+1}/{num_epochs}...")  

    for i, batch in enumerate(dataloader):
        print(f"Processing Batch {i+1}/{len(dataloader)}")  
        
        pixel_values = batch["pixel_values"].to(device)
        depth_targets = F.interpolate(batch["depth_map"].to(device), size=(384, 384), mode="bilinear", align_corners=False)

        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values).predicted_depth
        loss = criterion(outputs, depth_targets.squeeze(1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 10 == 0:  
            print(f"Epoch {epoch+1}, Batch {i+1}: Loss = {loss.item():.4f}")

    print(f"Epoch [{epoch+1}/{num_epochs}] Completed! Avg Loss: {total_loss/len(dataloader):.4f}")




model.save_pretrained("fine_tuned_dpt")
processor.save_pretrained("fine_tuned_dpt")
print(" Model saved successfully!")
