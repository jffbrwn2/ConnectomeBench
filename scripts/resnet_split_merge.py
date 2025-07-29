import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
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
            labels_file: Path to JSON file with unified result structure
            transform: Image transformations
            concatenate_images: If True, concatenate 3 images horizontally (1024x3072)
                               If False, stack as channels (3, 1024, 1024)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.concatenate_images = concatenate_images
        self.split_or_merge_correction = split_or_merge_correction
        
        # Load labels from unified JSON structure
        with open(labels_file, 'r') as f:
            annotations = json.load(f)

        # Filter annotations by task
        task_annotations = [x for x in annotations if x.get('task') == split_or_merge_correction]
        
        if split_or_merge_correction == "split_comparison":

            # For split comparison, we need pairs of neurons
            self.data = []
            self.labels = []
            
            # Group by operation_id to find pairs
      
            for ann in task_annotations:
                # Find the positive and negative examples
                positive_examples = ann.get('root_id_requires_split')
                negative_examples = ann.get('root_id_does_not_require_split')
                
                if (len(positive_examples) > 0 and len(negative_examples) > 0) and (positive_examples != negative_examples):
                    pos = positive_examples
                    neg = negative_examples
                    
                    # Randomly choose one ordering to avoid data leakage
                    if random.random() < 0.5:
                        # Positive example first
                        self.data.append((
                            pos,
                            neg, 
                            ann.get('merge_coords'),
                            ann.get('prompt_options')
                        ))
                        self.labels.append(0)  # Positive example first
                    else:
                        # Negative example first
                        self.data.append((
                            neg,
                            pos, 
                            ann.get('merge_coords'),
                            ann.get('prompt_options')
                        ))
                        self.labels.append(1)  # Negative example first
            
        elif split_or_merge_correction == "split_identification":

            
            # For split identification, each annotation is a single neuron
            self.data = []
            self.labels = []

            for ann in task_annotations:
                neuron_id = ann.get('id')
                merge_coords = ann.get('merge_coords')
                is_split = ann.get('is_split', False)
                
                if len(neuron_id) > 0 and len(merge_coords) > 0:
                    # Create folder extension for image path
                    folder_extension = f"{merge_coords}"
                    self.data.append((neuron_id, merge_coords, ann.get('prompt_options')))
                    self.labels.append(is_split)

        elif split_or_merge_correction == "merge_comparison":
            # For merge comparison, we need the prompt options
            # I have to update this code for the new format of the json file
            self.data = []
            self.labels = []
            
            # Group by operation_id
            operation_groups = {}
            for ann in task_annotations:
               
                prompt_options = ann.get('prompt_options', [])
                expected_choice_ids = ann.get('correct_answer', [])
                if len(prompt_options) < 2:
                    continue

                self.data.append(prompt_options)
                # Find the index of the correct choice
                if isinstance(expected_choice_ids, list) and expected_choice_ids:
                    correct_id = expected_choice_ids[0]
                    option_ids = [opt.get('id') for opt in prompt_options]
                    try:
                        label = option_ids.index(correct_id)
                    except ValueError:
                        raise ValueError(f"Correct ID {correct_id} not found in prompt options")
                self.labels.append(label)

                        
        elif split_or_merge_correction == "merge_identification":
            # For merge identification, each annotation is a single option
            self.data = []
            self.labels = []

            # Randomly permute annotations to avoid bias
            task_annotations = task_annotations.copy()
            random.shuffle(task_annotations)
            
            # Track operation_ids we've already seen to prevent data leakage
            seen_operation_ids = set()
            
            for ann in task_annotations:
                operation_id = ann.get('operation_id', 'unknown')
                
                # Only add if we haven't seen this operation_id before
                if operation_id not in seen_operation_ids:
                    # Check if this neuron is the correct choice
                    is_correct = ann.get("is_correct_merge", False)
                    self.data.append(ann)  # Store the full annotation for image paths
                    self.labels.append(is_correct)
                    
                    # Mark this operation_id as seen
                    seen_operation_ids.add(operation_id)


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

            root_id_requires_split, root_id_does_not_require_split, merge_coords, prompt_options = self.data[idx]

            label = self.labels[idx]
            label_idx = self.label_to_idx[label]
            images = []
            views = ['front', 'top', 'side']

            if label == 0:
                for prompt_option in prompt_options:
                    for view in views:
                        image = Image.open(prompt_option["paths"]["zoomed"][view]).convert('L')  # Convert to grayscale
                        images.append(np.array(image))
            else:
                for prompt_option in prompt_options[::-1]:
                    for view in views:
                        image = Image.open(prompt_option["paths"]["zoomed"][view]).convert('L')  # Convert to grayscale
                        images.append(np.array(image))



            # for prompt_option in prompt_options:
                
            #     for view in views:
            #         image = Image.open(prompt_option["paths"]["zoomed"][view]).convert('L')  # Convert to grayscale
            #         images.append(np.array(image))

            # root_id_requires_split, root_id_does_not_require_split, merge_coords = self.data[idx]

            # label = self.labels[idx]
            # label_idx = self.label_to_idx[label]
            
            # # Load three images for this mesh
            # mesh_folder = os.path.join(self.data_dir)
            # images = []
            # views = ['front', 'top', 'side']
            # if label == 0:
            #     for i in range(3):
            #         img_path = os.path.join(mesh_folder, f'base_{root_id_requires_split}_{root_id_does_not_require_split}_{merge_coords}/base_{root_id_requires_split}_zoomed_{views[i]}.png')
            #         image = Image.open(img_path).convert('L')  # Convert to grayscale
            #         images.append(np.array(image))
            #     for i in range(3):
            #         img_path = os.path.join(mesh_folder, f'base_{root_id_requires_split}_{root_id_does_not_require_split}_{merge_coords}/base_{root_id_does_not_require_split}_zoomed_{views[i]}.png')
            #         image = Image.open(img_path).convert('L')  # Convert to grayscale
            #         images.append(np.array(image))
            # else:
            #     for i in range(3):
            #         img_path = os.path.join(mesh_folder, f'base_{root_id_requires_split}_{root_id_does_not_require_split}_{merge_coords}/base_{root_id_does_not_require_split}_zoomed_{views[i]}.png')
            #         image = Image.open(img_path).convert('L')  # Convert to grayscale
            #         images.append(np.array(image))
            #     for i in range(3):
            #         img_path = os.path.join(mesh_folder, f'base_{root_id_requires_split}_{root_id_does_not_require_split}_{merge_coords}/base_{root_id_requires_split}_zoomed_{views[i]}.png')
            #         image = Image.open(img_path).convert('L')  # Convert to grayscale
            #         images.append(np.array(image))
                    
        elif self.split_or_merge_correction == "split_identification":
            root_id, merge_coords, prompt_options = self.data[idx]
            label = self.labels[idx]
            label_idx = self.label_to_idx[label]
            
            # Load three images for this mesh
            mesh_folder = os.path.join(self.data_dir)
            images = []
            views = ['front', 'top', 'side']
            for prompt_option in prompt_options:
                if prompt_option.get('id') == root_id:
                    for view in views:
                        image = Image.open(prompt_option["paths"]["zoomed"][view]).convert('L')  # Convert to grayscale
                        images.append(np.array(image))
                    break

                
        elif self.split_or_merge_correction == "merge_comparison":
            prompt_options = self.data[idx]
            label = self.labels[idx]
            label_idx = self.label_to_idx[label]
            images = []
            views = ['front', 'top', 'side']
            for prompt_option in prompt_options:
                for view in views:
                    image = Image.open(prompt_option["paths"]["zoomed"][view]).convert('RGB')  # Keep RGB colors
                    images.append(np.array(image))

        elif self.split_or_merge_correction == "merge_identification":
            annotation = self.data[idx]  # Full annotation from unified structure
            label = self.labels[idx]
            label_idx = self.label_to_idx[label]
            images = []
            views = ['front', 'top', 'side']

            # Find the option data for this neuron
            neuron_id = annotation.get('id')
            prompt_options = annotation.get('prompt_options', [])
            
            # Find the matching option
            option_data = None
            for opt in prompt_options:
                if opt.get('id') == neuron_id:
                    option_data = opt
                    break
            
            if option_data:
                for view in views:
                    image = Image.open(option_data["paths"]["zoomed"][view]).convert('RGB')
                    images.append(np.array(image))
            else:
                # Fallback: create empty RGB images if option not found
                for view in views:
                    images.append(np.zeros((1024, 1024, 3), dtype=np.uint8))

        # Force concatenation mode for merge tasks with RGB images
        use_concatenation = self.concatenate_images or self.split_or_merge_correction in ["merge_comparison", "merge_identification"]
        
        if use_concatenation:
            # Concatenate horizontally: 1024 x (N*1024)
            combined_image = np.concatenate(images, axis=1)
            combined_image = Image.fromarray(combined_image).convert('RGB')
        else:
            # Stack as channels: N x 1024 x 1024 (for split tasks)
            combined_image = np.stack(images, axis=0)
            combined_image = torch.from_numpy(combined_image).float()
            # Normalize to [0, 1]
            combined_image = combined_image / 255.0
        
        if use_concatenation:
            # For concatenated images (PIL Images), we need transforms that include ToTensor
            if self.transform:
                # Check if transform includes ToTensor by trying to find it in the transforms list
                has_to_tensor = False
                if hasattr(self.transform, 'transforms'):
                    for t in self.transform.transforms:
                        if 'ToTensor' in str(type(t)):
                            has_to_tensor = True
                            break
                
                if has_to_tensor:
                    combined_image = self.transform(combined_image)
                else:
                    # Transform doesn't include ToTensor, manually convert first
                    import torchvision.transforms as transforms
                    to_tensor = transforms.ToTensor()
                    combined_image = to_tensor(combined_image)
            else:
                # No transform, manually convert PIL to tensor
                import torchvision.transforms as transforms
                to_tensor = transforms.ToTensor()
                combined_image = to_tensor(combined_image)
        else:
            # For stacked images, already a tensor
            if self.transform:
                # Apply transforms to each channel separately if needed
                pass
        
        # Final safety check - ensure we always return a tensor
        if not isinstance(combined_image, torch.Tensor):
            import torchvision.transforms as transforms
            to_tensor = transforms.ToTensor()
            combined_image = to_tensor(combined_image)
            
        return combined_image, label_idx

# Custom ResNet for different input sizes
class ConnectomicsResNet(nn.Module):
    def __init__(self, num_classes=5, concatenate_images=True, pretrained=True, split_or_merge_correction=None):
        super(ConnectomicsResNet, self).__init__()
        
        # if concatenate_images:
        #     # For concatenated images (1024 x 3072)
        self.resnet = models.resnet50(pretrained=pretrained)
        # Modify first conv layer to handle different input size
        if split_or_merge_correction == "merge_comparison":
            # Merge comparison: RGB images concatenated horizontally = 3 channels
            self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif split_or_merge_correction == "split_comparison":
            # Split comparison: 2 neurons x 3 views, grayscale stacked = 6 channels
            self.resnet.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif split_or_merge_correction == "merge_identification":
            # Merge identification: RGB images concatenated horizontally = 3 channels
            self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif split_or_merge_correction == "split_identification":
            # Split identification: 3 views, grayscale stacked = 3 channels
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

# Training function for a single fold
def train_single_fold(train_dataset, val_dataset, full_dataset, fold_idx, data_dir, labels_file, 
                     split_or_merge_correction, concatenate_images=True, num_epochs=50, 
                     batch_size=16, learning_rate=1e-4, output_dir='scripts/output'):
    """Train model for a single fold"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training Fold {fold_idx + 1}/5 on device: {device}')
    
    # Get transforms
    train_transform, val_transform = get_transforms(concatenate_images)
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
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
        print(f'Fold {fold_idx + 1}, Epoch {epoch+1}/{num_epochs}')
        
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
        
        # Save best model for this fold
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            fold_model_path = os.path.join(output_dir, f'best_model_fold_{fold_idx + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'label_to_idx': full_dataset.label_to_idx,
                'fold_idx': fold_idx
            }, fold_model_path)
            print(f'New best model for fold {fold_idx + 1} saved with validation accuracy: {val_acc:.2f}%')
    
    # Plot training curves for this fold
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Fold {fold_idx + 1} - Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title(f'Fold {fold_idx + 1} - Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'training_curves_fold_{fold_idx + 1}.png'))
    plt.close()
    
    # Final evaluation for this fold
    print(f'\nFinal Evaluation on Validation Set - Fold {fold_idx + 1}:')
    print(classification_report(val_targets, val_preds, 
                              labels=list(range(full_dataset.num_classes)),
                              target_names=[str(full_dataset.idx_to_label[i]) for i in range(full_dataset.num_classes)]))
    
    # Confusion matrix for this fold
    cm = confusion_matrix(val_targets, val_preds, labels=list(range(full_dataset.num_classes)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[str(full_dataset.idx_to_label[i]) for i in range(full_dataset.num_classes)],
                yticklabels=[str(full_dataset.idx_to_label[i]) for i in range(full_dataset.num_classes)])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - Fold {fold_idx + 1}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_fold_{fold_idx + 1}.png'))
    plt.close()
    
    return model, best_val_acc, val_preds, val_targets, train_losses, val_losses

# Main training loop with cross validation
def train_model_cv(data_dir, labels_file, split_or_merge_correction, concatenate_images=True, 
                  num_epochs=50, batch_size=16, learning_rate=1e-4, n_folds=5, output_dir='scripts/output'):
    """Train model using k-fold cross validation"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create full dataset without transforms initially
    full_dataset = ConnectomicsDataset(data_dir, labels_file, split_or_merge_correction, 
                                     transform=None, concatenate_images=concatenate_images)
    
    # Get labels for stratified splitting
    labels = [full_dataset.labels[i] for i in range(len(full_dataset))]
    
    # Initialize cross validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Store results for each fold
    fold_results = []
    all_val_preds = []
    all_val_targets = []
    
    print(f'Starting {n_folds}-fold cross validation...')
    

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(range(len(full_dataset)), labels)):
        print(f'\n{"="*60}')
        print(f'Fold {fold_idx + 1}/{n_folds}')
        print(f'{"="*60}')
        
        # Create train and validation datasets for this fold
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
        
        # Train model for this fold
        model, val_acc, val_preds, val_targets, train_losses, val_losses = train_single_fold(
            train_dataset, val_dataset, full_dataset, fold_idx, data_dir, labels_file,
            split_or_merge_correction, concatenate_images, num_epochs, batch_size, 
            learning_rate, output_dir
        )
        
        # Store results
        fold_results.append({
            'fold_idx': fold_idx,
            'val_acc': val_acc,
            'val_preds': val_preds,
            'val_targets': val_targets,
            'train_losses': train_losses,
            'val_losses': val_losses
        })
        
        all_val_preds.extend(val_preds)
        all_val_targets.extend(val_targets)
        
        print(f'Fold {fold_idx + 1} completed with validation accuracy: {val_acc:.2f}%')
    
    # Calculate overall statistics
    fold_accuracies = [result['val_acc'] for result in fold_results]
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    best_fold_idx = np.argmax(fold_accuracies)
    best_fold_accuracy = fold_accuracies[best_fold_idx]
    
    print(f'\n{"="*60}')
    print(f'CROSS VALIDATION RESULTS')
    print(f'{"="*60}')
    print(f'Mean validation accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%')
    print(f'Individual fold accuracies: {[f"{acc:.2f}%" for acc in fold_accuracies]}')
    print(f'Best fold: {best_fold_idx + 1} with accuracy: {best_fold_accuracy:.2f}%')
    
    # Overall classification report
    print(f'\nOverall Classification Report:')
    print(classification_report(all_val_targets, all_val_preds, 
                              labels=list(range(full_dataset.num_classes)),
                              target_names=[str(full_dataset.idx_to_label[i]) for i in range(full_dataset.num_classes)]))
    
    # Overall confusion matrix
    cm = confusion_matrix(all_val_targets, all_val_preds, labels=list(range(full_dataset.num_classes)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[str(full_dataset.idx_to_label[i]) for i in range(full_dataset.num_classes)],
                yticklabels=[str(full_dataset.idx_to_label[i]) for i in range(full_dataset.num_classes)])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Overall Confusion Matrix (All Folds)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_confusion_matrix.png'))
    plt.show()
    
    # Plot fold accuracies
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, n_folds + 1), fold_accuracies, alpha=0.7, color='skyblue')
    plt.axhline(y=mean_acc, color='red', linestyle='--', label=f'Mean: {mean_acc:.2f}%')
    plt.fill_between(range(1, n_folds + 1), mean_acc - std_acc, mean_acc + std_acc, 
                     alpha=0.3, color='red', label=f'±1 std: {std_acc:.2f}%')
    plt.xlabel('Fold')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Cross Validation Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_validation_results.png'))
    plt.show()
    
    # Save cross validation results
    cv_results = {
        'fold_results': fold_results,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'all_val_preds': all_val_preds,
        'all_val_targets': all_val_targets,
        'label_to_idx': full_dataset.label_to_idx,
        'best_fold_idx': best_fold_idx,
        'best_fold_accuracy': best_fold_accuracy
    }
    
    torch.save(cv_results, os.path.join(output_dir, 'cross_validation_results.pth'))
    
    # Save a summary file with best fold accuracies
    summary_data = {
        'fold_accuracies': fold_accuracies,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'best_fold_idx': best_fold_idx,
        'best_fold_accuracy': best_fold_accuracy,
        'fold_details': []
    }
    
    for i, result in enumerate(fold_results):
        summary_data['fold_details'].append({
            'fold': i + 1,
            'best_validation_accuracy': result['val_acc'],
            'model_path': f'best_model_fold_{i + 1}.pth'
        })
    
    # Save summary as JSON for easy reading
    with open(os.path.join(output_dir, 'cross_validation_summary.json'), 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    # Also save as a simple text file for quick reference
    with open(os.path.join(output_dir, 'fold_accuracies.txt'), 'w') as f:
        f.write("Cross Validation Results Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Mean accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%\n")
        f.write(f"Best fold: {best_fold_idx + 1} with accuracy: {best_fold_accuracy:.2f}%\n\n")
        f.write("Individual fold accuracies:\n")
        for i, acc in enumerate(fold_accuracies):
            f.write(f"Fold {i + 1}: {acc:.2f}%\n")
    
    # Return the best model from the best performing fold
    best_model_path = os.path.join(output_dir, f'best_model_fold_{best_fold_idx + 1}.pth')
    best_checkpoint = torch.load(best_model_path)
    
    best_model = ConnectomicsResNet(num_classes=full_dataset.num_classes, 
                                  concatenate_images=concatenate_images,
                                  split_or_merge_correction=split_or_merge_correction).to(device)
    best_model.load_state_dict(best_checkpoint['model_state_dict'])
    
    print(f'\nBest model from fold {best_fold_idx + 1} with accuracy {best_fold_accuracy:.2f}%')
    
    return best_model, full_dataset.label_to_idx, cv_results

# Main training loop (original function - kept for backward compatibility)
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
                              target_names=[str(full_dataset.idx_to_label[i]) for i in range(full_dataset.num_classes)]))
    
    # Confusion matrix
    cm = confusion_matrix(val_targets, val_preds, labels=list(range(full_dataset.num_classes)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[str(full_dataset.idx_to_label[i]) for i in range(full_dataset.num_classes)],
                yticklabels=[str(full_dataset.idx_to_label[i]) for i in range(full_dataset.num_classes)])
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
    DATA_DIR = "output/mouse_split"
    # LABELS_FILE = "output/mouse_merge_2048nm/merge_identification_results_20250728_115852.json"
    LABELS_FILE = "output/mouse_merge_2048nm/merge_comparison_results_20250728_113748.json"
    # LABELS_FILE = "output/mouse_split/split_identification_results_20250728_123430.json"
    # LABELS_FILE = "output/mouse_split/split_comparison_results_20250728_121758.json"
    SPLIT_OR_MERGE_CORRECTION = "merge_comparison"
    # Choose approach: concatenate images or stack as channels
    CONCATENATE_IMAGES = False  # Set to False to use 3-channel stacking approach
    
    # Train the model using 5-fold cross validation
    print("Starting 5-fold cross validation training...")
    model, label_to_idx, cv_results = train_model_cv(
        data_dir=DATA_DIR,
        labels_file=LABELS_FILE,
        split_or_merge_correction=SPLIT_OR_MERGE_CORRECTION,
        concatenate_images=CONCATENATE_IMAGES,
        num_epochs=50,
        batch_size=16,  # Adjust based on your GPU memory
        learning_rate=1e-4,
        n_folds=5,
        output_dir='output'
    )
    
    # Print cross validation summary
    print(f"\nCross Validation Summary:")
    print(f"Mean accuracy: {cv_results['mean_accuracy']:.2f}% ± {cv_results['std_accuracy']:.2f}%")
    print(f"Individual fold accuracies: {[f'{acc:.2f}%' for acc in [result['val_acc'] for result in cv_results['fold_results']]]}")
    
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