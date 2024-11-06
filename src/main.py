import argparse
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
from model import EmotionDetecterCNN
from tqdm import tqdm

def train_val_split(data):
    train_data_size = int(0.8 * len(data))
    val_data_size = len(data) - train_data_size
    train_data, val_data = random_split(data, [train_data_size, val_data_size])
    return train_data, val_data

def init_dataloaders(train_data, val_data, test_data, train_sampler):
    batch_size = 32
    n_workers = 4

    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=n_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    return train_loader, val_loader, test_loader

def train(model, train_loader, val_loader, optimizer, criterion, scheduler, device, num_epochs):
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()

        with tqdm(total=len(train_loader), desc=f'Epoch: {epoch + 1}/{num_epochs}', position=0, leave=True) as pbar:
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
            avg_loss = evaluate(model, val_loader, criterion, device)
            scheduler.step(avg_loss)
        if (epoch + 1) % 5 == 0:
            torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                'EmotionClassifierCNN.ckpt')

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
            _, predictions = torch.max(logits, dim=1)
            num_correct += (predictions == labels).sum()
            num_samples += labels.size(0)
        
        accuracy = num_correct / num_samples
        avg_loss = total_loss / len(val_loader)
        print(f'Validation loss: {avg_loss}, accuracy: {accuracy}')
    return avg_loss
        
def test(model, test_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        num_correct = 0
        num_samples = 0

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predictions = torch.max(logits, dim=1)
            num_correct += (predictions == labels).sum()
            num_samples += labels.size(0)
        
        accuracy = num_correct / num_samples
        avg_loss = total_loss / len(test_loader)
        print(f'Test loss: {avg_loss}, accuracy: {accuracy}')

def calculate_mean_std(train_data):
    mean = 0
    std = 0
    samples = 0
    for inputs, _ in train_data:
        inputs = transforms.Grayscale(num_output_channels=1)(inputs)
        inputs = transforms.ToTensor()(inputs)
        mean += inputs.mean([1, 2])
        std += inputs.std([1, 2])
        samples += 1
    mean /= samples
    std /= samples  
    print(f'Mean: {mean}, Std: {std}')
    return mean, std
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data_dir = 'dataset/train'
    test_data_dir = 'dataset/test'
    train_dataset = datasets.ImageFolder(root=train_data_dir)
    train_data, val_data = train_val_split(train_dataset)
    
    mean, std = calculate_mean_std(train_data)
    
    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(3),
        transforms.RandomInvert(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        transforms.RandomResizedCrop(44, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_test_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_data.dataset.transform = train_transforms
    val_data.dataset.transform = val_test_transforms
    test_data = datasets.ImageFolder(root=test_data_dir, transform=val_test_transforms)

    class_counts = [0] * len(train_dataset.classes)
    for _, label in train_data:
        class_counts[label] += 1
    class_weights = 1 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[label] for _, label in train_data]
    train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader, val_loader, test_loader = init_dataloaders(train_data, val_data, test_data, train_sampler)

    model = EmotionDetecterCNN(num_classes=len(train_dataset.classes))

    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 20

    if args.test:
        checkpoint = torch.load('EmotionClassifierCNN.ckpt', map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        test(model, test_loader, criterion, device)
    else:
        train(model, train_loader, val_loader, optimizer, criterion, scheduler, device, num_epochs)
    
if __name__ == "__main__":
    main()
