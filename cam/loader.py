import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import torch.utils.data

def generate_loader(batch_size: int):
    img_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])

    train_data_dir = datasets.ImageFolder("/Users/alinatamkevich/images/", transform=img_transform)
    train_loader = torch.utils.data.DataLoader(train_data_dir, batch_size=batch_size, shuffle=True)
    return train_loader
