import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch import nn
from model import EmotionDetecterCNN
from tqdm import tqdm

def train_val_split(data):
    train_data_size = int(0.8 * len(data))
    val_data_size = len(data) - train_data_size
    train_data, val_data = random_split(data, [train_data_size, val_data_size])
    return train_data, val_data

def init_dataloaders(train_data, val_data, test_data):
    batch_size = 32
    n_workers = 4

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    return train_loader, val_loader, test_loader

def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs):
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()

        with tqdm(total=len(train_loader), desc=f'Epoch: {epoch + 1}/{num_epochs}', position = 0, leave=True) as pbar:
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                logits = model(inputs)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
            evaluate(model, val_loader, criterion, device)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        num_correct = 0
        num_samples = 0

        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            print(loss)
            print(logits)
            _, predictions = torch.max(logits, dim=1)
                               
def main():
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(3),
        transforms.RandomInvert(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.ToTensor()
    ])

    val_test_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    train_data_dir = 'dataset/train'
    test_data_dir = 'dataset/test'
    train_dataset = datasets.ImageFolder(root=train_data_dir)
    train_data, val_data = train_val_split(train_dataset)
    train_data.dataset.transform = train_transforms
    val_data.dataset.transform = val_test_transforms
    test_data = datasets.ImageFolder(root=test_data_dir, transform=val_test_transforms)

    train_loader, val_loader, test_loader = init_dataloaders(train_data, val_data, test_data)

    model = EmotionDetecterCNN(num_classes=len(train_dataset.classes))

    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs = 10

    train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs)
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict()},
                'model.ckpt')

if __name__ == "__main__":
    main()


