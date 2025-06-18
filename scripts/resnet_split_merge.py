import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from PIL import Image
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import json
import random
# Custom Dataset for Connectomics Data
class ConnectomicsDataset(Dataset):
    def __init__(self, data_dir, labels_file, split_or_merge_correction,transform=None, concatenate_images=True):
        """
        Dataset for connectomics mesh classification
        
        Args:
            data_dir: Directory containing mesh folders
            labels_file: Path to CSV file with proofread root id,current root id,species,xmin,ymin,zmin,xmax,ymax,zmax,unit,human answer 1,confidence 1 columns
            transform: Image transformations
            concatenate_images: If True, concatenate 3 images horizontally (1024x3072)
                               If False, stack as channels (3, 1024, 1024)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.concatenate_images = concatenate_images
        self.split_or_merge_correction = split_or_merge_correction
        # Load labels
        if split_or_merge_correction == "split_comparison":
            with open(labels_file, 'r') as f:
                annotations = json.load(f)

            self.data = [(x['root_id_requires_split'], x['root_id_does_not_require_split'], x['merge_coords']) for x in annotations]
            self.labels = [0] * len(self.data)
            self.labels[:len(self.labels)//2] = [1] * (len(self.labels)//2)
            random.shuffle(self.labels)
        elif split_or_merge_correction == "split_identification":
            with open(labels_file, 'r') as f:
                annotations = json.load(f)
            
            self.data = [(x['root_id_requires_split'], f"{"_".join([str(x) for x in sorted([x['root_id_requires_split'],x['root_id_does_not_require_split']])])}_{x['merge_coords']}") for x in annotations] +[(x['root_id_does_not_require_split'], f"{"_".join([str(x) for x in sorted([x['root_id_requires_split'],x['root_id_does_not_require_split']])])}_{x['merge_coords']}") for x in annotations]
            self.labels = [True] * len(self.data) + [False] * len(self.data)
        elif split_or_merge_correction == "merge_comparison":
            with open(labels_file, 'r') as f:
                annotations = json.load(f)

            self.data = [x['prompt_options'] for x in annotations if len(x['prompt_options']) == 2]
            self.labels = [x['options_presented_ids'].index(x['expected_choice_ids'][0]) for x in annotations if len(x['prompt_options']) == 2]
        elif split_or_merge_correction == "merge_identification":
            with open(labels_file, 'r') as f:
                annotations = json.load(f)

            self.data = [prompt_option for x in annotations for prompt_option in x['prompt_options']]
            self.labels = [x["options_presented_ids"].index(prompt_option["id"]) for x in annotations for prompt_option in x['prompt_options']] 
        else:
            raise ValueError(f"Invalid split_or_merge_correction: {split_or_merge_correction}")
        # Create label to index mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.split_or_merge_correction == "split_comparison":
            root_id_requires_split, root_id_does_not_require_split, merge_coords = self.data[idx]

            label = self.labels[idx]
            label_idx = self.label_to_idx[label]
            
            # Load three images for this mesh
            mesh_folder = os.path.join(self.data_dir)
            images = []
            views = ['front', 'top', 'side']
            if label == 0:
                for i in range(3):
                    img_path = os.path.join(mesh_folder, f'base_{"_".join([str(x) for x in sorted([root_id_requires_split,root_id_does_not_require_split])])}_{merge_coords}/base_{root_id_requires_split}_zoomed_{views[i]}.png')  # Adjust filename pattern
                    image = Image.open(img_path).convert('L')  # Convert to grayscale
                    images.append(np.array(image))
                for i in range(3):
                    img_path = os.path.join(mesh_folder, f'base_{"_".join([str(x) for x in sorted([root_id_requires_split,root_id_does_not_require_split])])}_{merge_coords}/base_{root_id_does_not_require_split}_zoomed_{views[i]}.png')  # Adjust filename pattern
                    image = Image.open(img_path).convert('L')  # Convert to grayscale
                    images.append(np.array(image))
            else:
                for i in range(3):
                    img_path = os.path.join(mesh_folder, f'base_{"_".join([str(x) for x in sorted([root_id_requires_split,root_id_does_not_require_split])])}_{merge_coords}/base_{root_id_does_not_require_split}_zoomed_{views[i]}.png')  # Adjust filename pattern
                    image = Image.open(img_path).convert('L')  # Convert to grayscale
                    images.append(np.array(image))
                for i in range(3):
                    img_path = os.path.join(mesh_folder, f'base_{"_".join([str(x) for x in sorted([root_id_requires_split,root_id_does_not_require_split])])}_{merge_coords}/base_{root_id_requires_split}_zoomed_{views[i]}.png')  # Adjust filename pattern
                    image = Image.open(img_path).convert('L')  # Convert to grayscale
                    images.append(np.array(image))
        elif self.split_or_merge_correction == "split_identification":
            root_id, folder_extension = self.data[idx]
            label = self.labels[idx]
            label_idx = self.label_to_idx[label]
            
            # Load three images for this mesh
            mesh_folder = os.path.join(self.data_dir)
            images = []
            views = ['front', 'top', 'side']
            for i in range(3):
                img_path = os.path.join(mesh_folder, f'base_{folder_extension}/base_{root_id}_zoomed_{views[i]}.png')  # Adjust filename pattern
                image = Image.open(img_path).convert('L')  # Convert to grayscale
                images.append(np.array(image))
        elif self.split_or_merge_correction == "merge_comparison":
            prompt_options = self.data[idx]
            label = self.labels[idx]
            label_idx = self.label_to_idx[label]
            images = []
            views = ['front', 'top', 'side']
            for prompt_option in prompt_options:
                for view in views:


                    image = Image.open(prompt_option["paths"]["zoomed"][view]).convert('L')  # Convert to grayscale
                    images.append(np.array(image))

        elif self.split_or_merge_correction == "merge_identification":
            prompt_option_paths = self.data[idx]
            label = self.labels[idx]
            label_idx = self.label_to_idx[label]
            images = []
            views = ['front', 'top', 'side']

            for view in views:
                image = Image.open(prompt_option_paths["paths"]["zoomed"][view]).convert('L')
                images.append(np.array(image))


            
        if self.concatenate_images:
            # Concatenate horizontally: 1024 x 3072
            combined_image = np.concatenate(images, axis=1)
            combined_image = Image.fromarray(combined_image).convert('RGB')
        else:
            # Stack as channels: 3 x 1024 x 1024
            combined_image = np.stack(images, axis=0)
            combined_image = torch.from_numpy(combined_image).float()
            # Normalize to [0, 1]
            combined_image = combined_image / 255.0
        
        if self.transform and self.concatenate_images:
            combined_image = self.transform(combined_image)
        elif not self.concatenate_images and self.transform:
            # Apply transforms to each channel separately if needed
            pass
            
        return combined_image, label_idx

# Custom ResNet for different input sizes
class ConnectomicsResNet(nn.Module):
    def __init__(self, num_classes=5, concatenate_images=True, pretrained=True, split_or_merge_correction=None):
        super(ConnectomicsResNet, self).__init__()
        
        # if concatenate_images:
        #     # For concatenated images (1024 x 3072)
        self.resnet = models.resnet50(pretrained=pretrained)
        # Modify first conv layer to handle different input size
        if split_or_merge_correction == "merge_comparison" or split_or_merge_correction == "split_comparison":
            self.resnet.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif split_or_merge_correction == "split_identification" or split_or_merge_correction == "merge_identification":
            self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else: 
            raise ValueError(f"Invalid split_or_merge_correction: {self.split_or_merge_correction}")
        # Add adaptive pooling to handle non-standard input size
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # else:
        #     # For stacked images (3 x 1024 x 1024)
        #     self.resnet = models.resnet50(pretrained=pretrained)
        #     # Keep original architecture but modify for 3-channel input
        #     self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Replace final layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.resnet(x)

# Data augmentation and preprocessing
def get_transforms(concatenate_images=True):
    if concatenate_images:
        train_transform = transforms.Compose([
            transforms.Resize((512, 1536)),  # Resize to manageable size
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((512, 1536)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # For stacked channel approach
        train_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((512, 512)),
        ])
    
    return train_transform, val_transform

# Handle class imbalance
def get_weighted_sampler(dataset):
    """Create weighted sampler for imbalanced dataset"""
    # Handle both full datasets and subsets
    if hasattr(dataset, 'indices'):  # This is a Subset
        # Get labels for the subset indices
        labels = [dataset.dataset.labels[dataset.indices[i]] for i in range(len(dataset))]
    else:  # This is the full dataset
        labels = [dataset.labels[i] for i in range(len(dataset))]
    
    class_counts = Counter(labels)
    
    # Calculate weights
    total_samples = len(labels)
    weights = {label: total_samples / count for label, count in class_counts.items()}
    
    # Create sample weights
    sample_weights = [weights[label] for label in labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

def get_class_weights(dataset):
    """Calculate class weights for loss function"""
    labels = [dataset.labels[i] for i in range(len(dataset))]
    unique_labels = sorted(set(labels))
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.array(unique_labels),
        y=np.array(labels)
    )
    
    return torch.FloatTensor(class_weights)

# Training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

# Validation function
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_targets

# Main training loop
def train_model(data_dir, labels_file, split_or_merge_correction, concatenate_images=True, num_epochs=50, batch_size=16, learning_rate=1e-4):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Get transforms
    train_transform, val_transform = get_transforms(concatenate_images)
    
    # Create datasets
    full_dataset = ConnectomicsDataset(data_dir, labels_file, split_or_merge_correction, transform=train_transform, 
                                     concatenate_images=concatenate_images)

    # Split dataset (80-20 train-val split)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Apply validation transforms to val dataset
    val_dataset.dataset.transform = val_transform

    # Create weighted sampler for training
    weighted_sampler = get_weighted_sampler(train_dataset)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=weighted_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = ConnectomicsResNet(num_classes=full_dataset.num_classes, 
                             concatenate_images=concatenate_images,
                             split_or_merge_correction=split_or_merge_correction).to(device)
    
    # Loss function with class weights
    class_weights = get_class_weights(full_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer with different learning rates for different parts
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'fc' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': learning_rate * 0.1},  # Lower LR for pretrained layers
        {'params': classifier_params, 'lr': learning_rate}       # Higher LR for new classifier
    ], weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                   patience=5)
    
    # Training loop
    best_val_acc = 0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_targets = validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'label_to_idx': full_dataset.label_to_idx
            }, 'scripts/output/best_connectomics_model.pth')
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('scripts/output/training_curves.png')
    plt.show()
    
    # Final evaluation
    print('\nFinal Evaluation on Validation Set:')
    print(classification_report(val_targets, val_preds, 
                              labels=list(range(full_dataset.num_classes)),
                              target_names=[full_dataset.idx_to_label[i] for i in range(full_dataset.num_classes)]))
    
    # Confusion matrix
    cm = confusion_matrix(val_targets, val_preds, labels=list(range(full_dataset.num_classes)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[full_dataset.idx_to_label[i] for i in range(full_dataset.num_classes)],
                yticklabels=[full_dataset.idx_to_label[i] for i in range(full_dataset.num_classes)])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('scripts/output/confusion_matrix.png')
    plt.show()
    
    return model, full_dataset.label_to_idx

# Inference function
def predict_mesh(model, data_dir, proofread_root_id, current_root_id, label_to_idx, device, concatenate_images=True):
    """Predict class for a single mesh"""
    model.eval()
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    # Load three images
    images = []
    views = ['front', 'top', 'side']
    for i in range(3):
        img_path = os.path.join(data_dir, f'{proofread_root_id}_{current_root_id}_{views[i]}.png')
        image = Image.open(img_path).convert('L')
        images.append(np.array(image))
    
    if concatenate_images:
        combined_image = np.concatenate(images, axis=1)
        combined_image = Image.fromarray(combined_image).convert('RGB')
        
        # Use full 1024x3072 resolution to match training without transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(combined_image).unsqueeze(0).to(device)
    else:
        combined_image = np.stack(images, axis=0)
        combined_image = torch.from_numpy(combined_image).float() / 255.0
        
        # Use full 1024x1024 resolution to match training
        input_tensor = combined_image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        predicted_class = idx_to_label[predicted.item()]
        confidence = probabilities[0][predicted].item()
    
    return predicted_class, confidence, probabilities[0].cpu().numpy()

# Example usage
if __name__ == "__main__":
    # Set up your data paths
    DATA_DIR = "scripts/output/mouse_split"
    LABELS_FILE = "scripts/output/mouse_split/split_comparison_results_20250617_114754.json"
    SPLIT_OR_MERGE_CORRECTION = "split_comparison"
    # Choose approach: concatenate images or stack as channels
    CONCATENATE_IMAGES = False  # Set to False to use 3-channel stacking approach
    
    # Train the model
    model, label_to_idx = train_model(
        data_dir=DATA_DIR,
        labels_file=LABELS_FILE,
        split_or_merge_correction=SPLIT_OR_MERGE_CORRECTION,
        concatenate_images=CONCATENATE_IMAGES,
        num_epochs=50,
        batch_size=16,  # Adjust based on your GPU memory
        learning_rate=1e-4
    )
    
    # Example prediction
    # data_dir = "scripts/output/mouse_segment_classification_full"
    # proofread_root_id = "123456"  # Example proofread root id
    # current_root_id = "789012"    # Example current root id
    # predicted_class, confidence, all_probs = predict_mesh(
    #     model, data_dir, proofread_root_id, current_root_id, label_to_idx, 
    #     torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    #     concatenate_images=CONCATENATE_IMAGES
    # )
    # print(f"Predicted class: {predicted_class}, Confidence: {confidence:.4f}")