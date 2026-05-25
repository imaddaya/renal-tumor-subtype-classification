from torch.utils.data import DataLoader
from dataset_loader import WSIDataset, get_default_transform

csv_path = "/home/hpdeadman/Grad_Project/data/wsi_metadata.csv"

dataset = WSIDataset(
    csv_path=csv_path,
    split="train",
    num_patches=8,
    transform=get_default_transform()
)

loader = DataLoader(dataset, batch_size=2, shuffle=True)

images, labels, wsi_ids = next(iter(loader))

print("images shape:", images.shape)
print("labels:", labels)
print("wsi_ids:", wsi_ids)