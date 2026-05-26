import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import fire  # pip install fire
from huggingface_hub import hf_hub_download
import wandb  # optional, for logging
import torchvision.transforms.functional as TF
import random
from copy import deepcopy
from evalutil import distort_images
from wartermark import load_model
import time

# TODO: CHECK whether the resolution of the training images plays a role? If we train on lower-rase GPT-B images and test on GPT-L images, does it still work?



class ImageTokenDataset(Dataset):
    """Dataset that loads images and extracts original tokens from metadata."""
    
    def __init__(self, image_dir, transform=None, perturbation_transform=None):
        self.image_dir = Path(image_dir)
        self.image_paths = list(self.image_dir.glob("*.png"))
        self.transform = transform
        self.perturbation_transform = perturbation_transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Extract metadata containing original tokens
        # PNG metadata can be stored in different ways, adjust based on your format
        metadata = image.info
        
        # Assuming tokens are stored as a JSON string in metadata
        # Adjust this based on your actual metadata format
        if 'tokens' in metadata:
            original_tokens = torch.tensor(json.loads(metadata['tokens']), dtype=torch.long)
        elif 'original_tokens' in metadata:
            original_tokens = torch.tensor(json.loads(metadata['original_tokens']), dtype=torch.long)
        else:
            # Alternative: tokens might be in filename or separate file
            raise ValueError(f"No token metadata found in {img_path}")
        
        # Apply base transforms
        if self.transform:
            image = self.transform(image)
        
        # Apply perturbations for robustness
        if self.perturbation_transform:
            image = self.perturbation_transform(image)

        image = image.clamp(0, 1)  # Ensure image is in [0, 1] range
            
        return image, original_tokens, str(img_path)


class ImagePerturbations:
    """Collection of image perturbations for robustness training."""
    
    def __init__(self, 
                 gaussian_noise_std=0.1,
                 salt_pepper_prob=0.05,
                 jpeg_quality_range=(30, 95),
                 blur_kernel_size=11,
                 apply_prob=0.8,
                 brightness_factor=4.0,
                    contrast_factor=2.0,
                    hue_factor=0.1,
                    saturation_factor=2.0,
                 strength_multiplier=1.0):
        self.base_gaussian_noise_std = gaussian_noise_std
        self.base_salt_pepper_prob = salt_pepper_prob
        self.jpeg_quality_range = jpeg_quality_range
        self.blur_kernel_size = blur_kernel_size
        self.base_apply_prob = apply_prob
        self.strength_multiplier = strength_multiplier
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.hue_factor = hue_factor
        self.saturation_factor = saturation_factor
    
    def set_strength(self, strength_multiplier):
        """Update the strength multiplier for all perturbations."""
        self.strength_multiplier = strength_multiplier
    
    @property
    def gaussian_noise_std(self):
        return self.base_gaussian_noise_std * self.strength_multiplier
    
    @property
    def salt_pepper_prob(self):
        return min(self.base_salt_pepper_prob * self.strength_multiplier, 0.5)
    
    @property
    def apply_prob(self):
        return min(self.base_apply_prob, 1.0)
    
    def __call__(self, image):
        """Apply random perturbations to image tensor."""
        if torch.rand(1).item() > self.apply_prob:
            return image
        
        # Randomly choose perturbation type
        perturbation_type = torch.randint(0, 8, (1,)).item()
        
        random_strength = torch.rand(1).item() if torch.rand(1).item() < 0.5 else 1.0
        if perturbation_type == 0:
            # Gaussian noise
            noise = torch.randn_like(image) * self.gaussian_noise_std * random_strength
            image = torch.clamp(image + noise, 0, 1)
            
        elif perturbation_type == 1:
            # Salt and pepper noise
            mask = torch.rand_like(image)
            image = torch.where(mask < (self.salt_pepper_prob * random_strength)/2, torch.zeros_like(image), image)
            image = torch.where(mask > 1 - (self.salt_pepper_prob * random_strength)/2, torch.ones_like(image), image)
            
        elif perturbation_type == 2:
            # Gaussian blur (strength increases with kernel size)
            kernel_size = int(self.blur_kernel_size * self.strength_multiplier * random_strength)
            # Ensure kernel size is odd
            kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
            image = transforms.functional.gaussian_blur(image, kernel_size=kernel_size)
            
        elif perturbation_type == 3:
            # JPEG compression (simulate by adding compression artifacts)
            quality_min, quality_max = self.jpeg_quality_range
            quality = quality_max - self.strength_multiplier * random_strength * (quality_max - quality_min)
            quality = int(round(quality))
            # Add some quantization noise to simulate JPEG artifacts
            quantization_noise = torch.randn_like(image) * (100 - quality) / 1000
            image = torch.clamp(image + quantization_noise, 0, 1)

        elif perturbation_type == 4:
            # Random brightness adjustment
            brightness_factor = self.brightness_factor * self.strength_multiplier
            image = distort_images(image, brightness_factor=brightness_factor)

        elif perturbation_type == 5:
            # Random contrast adjustment
            contrast_factor = self.contrast_factor * self.strength_multiplier
            image = distort_images(image, contrast_factor=contrast_factor)

        elif perturbation_type == 6:
            # Random hue adjustment
            hue_factor = self.hue_factor * self.strength_multiplier
            image = distort_images(image, hue_factor=hue_factor)

        elif perturbation_type == 7:
            # Random saturation adjustment
            saturation_factor = self.saturation_factor * self.strength_multiplier
            image = distort_images(image, saturation_factor=saturation_factor)

        if DEBUG:
            pngimg = TF.to_pil_image(image.clamp(0, 1))
            pngimg.save("test.png")
            
        return image


class PerturbationScheduler:
    """Scheduler for gradually increasing perturbation strength during training."""
    
    def __init__(self, 
                 start_strength=0.1,
                 end_strength=1.0,
                 warmup_epochs=0,
                 cooldown_epochs=5,
                 schedule_type='linear'):
        """
        Args:
            start_strength: Initial perturbation strength multiplier
            end_strength: Final perturbation strength multiplier
            warmup_epochs: Number of epochs before starting to increase strength
            schedule_type: Type of schedule ('linear', 'cosine', 'exponential', 'step')
        """
        self.start_strength = start_strength
        self.end_strength = end_strength
        self.warmup_epochs = warmup_epochs
        self.cooldown_epochs = cooldown_epochs
        self.schedule_type = schedule_type
        
    def get_strength(self, epoch, total_epochs):
        """Get perturbation strength for current epoch."""
        if epoch < self.warmup_epochs:
            return self.start_strength
        elif epoch >= total_epochs - self.cooldown_epochs:
            # During cooldown, keep strength at end_strength
            return self.end_strength
        
        # Calculate progress after warmup
        progress = (epoch - self.warmup_epochs) / max(1, (total_epochs - self.warmup_epochs - self.cooldown_epochs))
        progress = min(1.0, progress)
        
        if self.schedule_type == 'linear':
            strength = self.start_strength + (self.end_strength - self.start_strength) * progress
            
        elif self.schedule_type == 'cosine':
            # Cosine annealing from start to end
            strength = self.end_strength - (self.end_strength - self.start_strength) * \
                      (1 + np.cos(np.pi * progress)) / 2
                      
        elif self.schedule_type == 'exponential':
            # Exponential growth
            strength = self.start_strength * (self.end_strength / self.start_strength) ** progress
            
        elif self.schedule_type == 'step':
            # Step increases every 25% of remaining epochs
            steps = int(progress * 4)
            strength = self.start_strength + (self.end_strength - self.start_strength) * steps / 4
            
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
            
        return strength
    

class ClusterClassifier(nn.Module):
    """Simple classifier to predict cluster indices from encoded features."""
    
    def __init__(self, vqencoder, mapping=None, dim=256, modelname="rar_xl", num_clusters=-1):
        """
        Args:
            vqencoder: VQ encoder model to extract features
            clusters: Precomputed cluster assignments for every token in the vocabulary
        """
        super(ClusterClassifier, self).__init__()
        self.vqmodel = vqencoder
        # initialize a classifier mapping from dimension of the vqencoder output to the number of clusters
        device = next(self.vqmodel.parameters()).device
        self.mapping = mapping.to(device) if mapping is not None else None
        self.dim = dim
        self.num_clusters = self.mapping.max().item() + 1 if num_clusters < 0 else num_clusters
        self.classifier = nn.Linear(self.dim, self.num_clusters).to(device)
        self.modelname = modelname

    def forward(self, x, tokens=None, return_logits=False):   # VQ Encoder expects range between 0 and 1 
        device = next(self.vqmodel.parameters()).device
        x = x.to(device)
        if self.modelname.startswith("GPT"):
            x = x * 2 - 1  # GPT models expect input in range [-1, 1]
        features = self.vqmodel(x)
        features = features.view(features.size(0), features.size(1), -1).transpose(1, 2) 
        logits = self.classifier(features)
        preds = torch.argmax(logits, dim=-1)
        if return_logits:
            preds = logits

        if tokens is None:
            return preds
        else:
            loss = F.cross_entropy(logits.transpose(1, 2), self.mapping[tokens].long())
            accuracy = (preds == self.mapping[tokens].long()).float().mean(-1)
        
            return (loss, accuracy), preds

    @classmethod
    def from_pretrained(cls, path, vqencoder=None, mapping=None, dim=256, modelname="rar_xl"):
        """Load a pre-trained cluster classifier."""
        cluster_classifier = cls(vqencoder, mapping=mapping, dim=dim, modelname=modelname)
        checkpoint = torch.load(Path(path), map_location="cpu")
        statedict = checkpoint["classifier_state_dict"]
        statedict = {(k.replace("vqencoder.", "vqmodel.") if k.startswith("vqencoder.") else k): v for k, v in statedict.items()}  # fix the state dict keys
        cluster_classifier.load_state_dict(statedict)
        cluster_classifier.eval()
        return cluster_classifier


class LatentMatcher(nn.Module):
    """Simple classifier to predict cluster indices from encoded features."""
    
    def __init__(self, vqmodel, mapping=None, dim=256):
        super(LatentMatcher, self).__init__()
        self.vqmodel = vqmodel
        device = next(vqmodel.parameters()).device
        self.mapping = mapping.to(device)
        self.dim = dim
        # self.num_clusters = self.mapping.max().item() + 1
        # self.classifier = nn.Linear(self.dim, self.num_clusters).to(device)

    def forward(self, x, tokens):   # VQ Encoder expects range between 0 and 1 
        features = self.vqmodel.encoder(x)
        features = features.view(features.size(0), features.size(1), -1).transpose(1, 2) 

        with torch.no_grad():
            targets = self.vqmodel.quantize.embedding(tokens).to(features.device).detach()
            preds = self.vqmodel.encode(x).detach()
        
        loss = F.mse_loss(features, targets)
        accuracy = (preds == tokens).float().mean(-1)
        
        return (loss, accuracy), preds


def train_encoder(
    vq_model,
    train_loader,
    val_loader=None,
    num_epochs=100,
    learning_rate=1e-4,
    clusters=None,
    device='cuda',
    save_dir='./checkpoints',
    log_wandb=False,
    perturbation_scheduler=None,
    perturbations=None,
    mode="clusterpred", # "clusterpred" or "latentmatch"
):
    """Train only the encoder of the VQVAE model."""
    if mode == "clusterpred":
        cluster_classifier = ClusterClassifier(vq_model.encoder, clusters)
    elif mode == "latentmatch":
        cluster_classifier = LatentMatcher(vq_model, clusters)
    
    # Freeze all parameters except encoder
    for param in cluster_classifier.parameters():
        param.requires_grad = True
    
    # Setup optimizer (only for encoder parameters)
    optimizer = torch.optim.AdamW(
        cluster_classifier.parameters(), 
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=num_epochs, eta_min=1e-6
    # )
    scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=1.0, total_iters=num_epochs
    )

    clusters = clusters.to(device) if clusters is not None else None
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    for epoch in range(num_epochs):
        # Update perturbation strength
        current_strength = 1.0  # Default if no scheduler
        if perturbation_scheduler and perturbations:
            current_strength = perturbation_scheduler.get_strength(epoch, num_epochs)
            perturbations.set_strength(current_strength)
            print(f'Epoch {epoch+1}: Perturbation strength = {current_strength:.3f}')
        
        # Training phase
        cluster_classifier.train()
        train_loss = 0
        train_accuracy = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (images, original_tokens, paths) in enumerate(pbar):
            images = images.to(device)
            original_tokens = original_tokens.to(device)
            # z_original = vq_model.quantize.embedding(original_tokens).to(device)
            
            optimizer.zero_grad()

            (loss, accuracy), preds = cluster_classifier(images, original_tokens)

            accuracy = accuracy.mean()  # Average over batch
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(cluster_classifier.vqmodel.parameters(), 1.0)
            
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_accuracy += accuracy.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy.item():.4f}',
                'pert': f'{current_strength:.2f}' if perturbation_scheduler else 'N/A',
                "lr": f'{optimizer.param_groups[0]["lr"]:.6f}',
            })
            
            # Log to wandb
            if log_wandb and batch_idx % 10 == 0:
                log_dict = {
                    'train/loss': loss.item(),
                    'train/accuracy': accuracy.item(),
                    'train/lr': optimizer.param_groups[0]['lr']
                }
                if perturbation_scheduler:
                    log_dict['train/perturbation_strength'] = current_strength
                wandb.log(log_dict)
        
        # Calculate epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_train_accuracy = train_accuracy / len(train_loader)
        
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_accuracy:.4f}')
        
        # Validation phase
        if val_loader is not None:
            cluster_classifier.eval()
            val_loss = 0
            val_accuracy = 0
            
            with torch.no_grad():
                for images, original_tokens, _ in tqdm(val_loader, desc='Validation'):
                    images = images.to(device)
                    original_tokens = original_tokens.to(device)
                    
                    (loss, accuracy), preds = cluster_classifier(images, original_tokens)

                    val_loss += loss.item()
                    val_accuracy += accuracy.item()
            
            avg_val_loss = val_loss / len(val_loader)
            avg_val_accuracy = val_accuracy / len(val_loader)
            
            print(f'Validation Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_accuracy:.4f}')
            
            if log_wandb:
                wandb.log({
                    'val/loss': avg_val_loss,
                    'val/accuracy': avg_val_accuracy,
                    'epoch': epoch
                })
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % 2 == 0:
            checkpoint = {
                'epoch': epoch,
                'classifier_state_dict': cluster_classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'train_accuracy': avg_train_accuracy
            }
            if val_loader is not None:
                checkpoint['val_loss'] = avg_val_loss
                checkpoint['val_accuracy'] = avg_val_accuracy
            if perturbation_scheduler:
                checkpoint['perturbation_strength'] = current_strength
                
            torch.save(checkpoint, save_dir / f'encoder_epoch_{epoch+1}.pt')
            print(f'Saved checkpoint to {save_dir}/encoder_epoch_{epoch+1}.pt')


DATADIR = "experiments_v2.2/i_gen_wm_pp_v2.2_100000samples_GPT-B_c2i_wm_clusters64_penalty1e-13or14_greenfrac=0.5"
DATADIR = "experiments_v1/gen_clean_v1_100000samples_rar_xl"
DEBUG = True
DEBUG = False


def train(
    data_dir=DATADIR,
    val_split=0.1,
    batch_size=32,
    num_epochs=30,
    learning_rate=1e-4,
    num_workers=0 if DEBUG else 8,
    device=0,
    save_subdir='checkpoints-encoder',
    use_wandb=False,
    wandb_project='vqvae-encoder-training',
    modelname="rar_xl",
    num_clusters=64, 
    load_clusters="clusters_balanced/balanced_kmeans_16.pt",
    # Perturbation parameters
    gaussian_noise_std=0.1,
    salt_pepper_prob=0.07,
    jpeg_quality_min=20,
    jpeg_quality_max=80,
    brightness_factor=4.0,
    contrast_factor=2.0,
    hue_factor=0.1,
    saturation_factor=2.0,
    blur_kernel_size=9,
    perturbation_prob=0.8,
    # Perturbation schedule parameters
    perturbation_start_strength=0.2 if not DEBUG else 1.,
    perturbation_end_strength=1.0,
    perturbation_warmup_epochs=0,
    perturbation_cooldown_epochs=15,
    perturbation_schedule='linear',
    # mode="clusterpred",  # "clusterpred" or "latentmatch"
    mode="clusterpred",
):
    """Train VQVAE Encoder with robustness to image perturbations."""
    
    # Initialize wandb if requested
    if use_wandb:
        wandb.init(project=wandb_project, config=locals())
    
    # Load VQ model (using your existing function structure)
    device = torch.device("cuda", device) if isinstance(device, int) else device
    
    # Create model (adjust based on your actual model imports)
    
    t = time.time()
    print("loading model...")
    vq_model = load_model(modelsize=modelname, onlyvae=True, device=device)
    t = time.time() - t
    print(f"model loaded in {t:.2f} seconds.")
    
    # Setup data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Setup perturbations
    perturbations = ImagePerturbations(
        gaussian_noise_std=gaussian_noise_std,
        salt_pepper_prob=salt_pepper_prob,
        jpeg_quality_range=(jpeg_quality_min, jpeg_quality_max),
        blur_kernel_size=blur_kernel_size,
        apply_prob=perturbation_prob,
        strength_multiplier=perturbation_start_strength,  # Start with low strength
        brightness_factor=brightness_factor,
        contrast_factor=contrast_factor,
        hue_factor=hue_factor,
        saturation_factor=saturation_factor,
    )
    
    # Setup perturbation scheduler
    perturbation_scheduler = PerturbationScheduler(
        start_strength=perturbation_start_strength,
        end_strength=perturbation_end_strength,
        warmup_epochs=perturbation_warmup_epochs,
        cooldown_epochs=perturbation_cooldown_epochs,
        schedule_type=perturbation_schedule
    )
    
    save_dir = Path(data_dir) / (save_subdir + ("_cp" if mode == "clusterpred" else "_lm") + f"_{num_epochs}epochs")
    print("Saving checkpoints to:", save_dir)
    
    # Create dataset
    dataset = ImageTokenDataset(
        Path(data_dir) / "images",
        transform=transform,
        perturbation_transform=perturbations
    )

    # create clusters of VQ vectors
    if num_clusters > 0:
        # load precomputed clusters
        clusterfile = Path(load_clusters)
        print("loading clusters from", clusterfile)
        vq_clusters = torch.load(clusterfile, map_location="cpu")
        # compute cluster sizes:
        clustersizes = (vq_clusters[:, None] == torch.arange(num_clusters, device=vq_clusters.device)[None, :]).sum(0)
        print("Max and min cluster sizes: ", clustersizes.max(), clustersizes.min())
        print("Num tokens:", vq_clusters.shape)
    else:
        vq_clusters = None
    
    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    val_dataset, train_dataset = torch.utils.data.random_split(
        dataset, [val_size, train_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    ) if val_size > 0 else None
    
    print(f"Dataset size: {len(dataset)} images")
    print(f"Train size: {train_size}, Validation size: {val_size}")
    print(f"Perturbation schedule: {perturbation_schedule} from {perturbation_start_strength} to {perturbation_end_strength}")
    print(f"Warmup epochs: {perturbation_warmup_epochs}")
    
    # Train encoder
    train_encoder(
        vq_model,
        train_loader,
        val_loader,
        clusters=vq_clusters,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
        save_dir=save_dir,
        log_wandb=use_wandb,
        perturbation_scheduler=perturbation_scheduler,
        perturbations=perturbations,
        mode=mode,
    )


def visualize_schedule(
    num_epochs=100, 
    perturbation_start_strength=0.1, 
    perturbation_end_strength=1.0, 
    perturbation_warmup_epochs=10,
    perturbation_schedule='linear'
):
    """
    Visualize the perturbation strength schedule.
    
    Example:
        python train_encoder.py visualize_schedule --num_epochs=100 --perturbation_schedule='cosine'
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return
    
    scheduler = PerturbationScheduler(
        start_strength=perturbation_start_strength,
        end_strength=perturbation_end_strength,
        warmup_epochs=perturbation_warmup_epochs,
        schedule_type=perturbation_schedule
    )
    
    epochs = list(range(num_epochs))
    strengths = [scheduler.get_strength(e, num_epochs) for e in epochs]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, strengths, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Perturbation Strength')
    plt.title(f'Perturbation Schedule: {perturbation_schedule}')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=perturbation_warmup_epochs, color='r', linestyle='--', 
               alpha=0.5, label=f'Warmup ends (epoch {perturbation_warmup_epochs})')
    plt.legend()
    plt.tight_layout()
    plt.savefig('perturbation_schedule.png', dpi=150)
    plt.show()
    print(f"Schedule plot saved to perturbation_schedule.png")


def test_dataloader(
    data_dir=DATADIR, 
    batch_size=16, 
    show_perturbations=True,
):
    """
    Test the data loader and show sample batch information.
    
    Example:
        python train_encoder.py test_dataloader --data_dir='path/to/images' --batch_size=4
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    if show_perturbations:
        perturbations = ImagePerturbations(
            gaussian_noise_std=0.1,
            salt_pepper_prob=0.05,
            jpeg_quality_range=(30, 95),
            blur_kernel_size=11,
            apply_prob=1.0,  # Always apply for visualization
            strength_multiplier=1.0
        )
    else:
        perturbations = None
    
    dataset = ImageTokenDataset(Path(data_dir) / "images", transform=transform, 
                              perturbation_transform=perturbations)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Total images found: {len(dataset)}")
    
    for i, (images, tokens, paths) in enumerate(loader):
        print(f"\nBatch {i+1}:")
        print(f"  Image shape: {images.shape}")
        print(f"  Token shape: {tokens.shape}")
        print(f"  Image value range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Sample paths: {paths[:2]}")
        if i >= 2:  # Show only first 3 batches
            break


def evaluate(
    checkpoint, 
    data_dir=DATADIR, 
    batch_size=32, 
    image_size=256, 
    device='cuda', 
    vq_model='VQ-16', 
    codebook_size=16384, 
    codebook_embed_dim=8,
    test_perturbations=True, 
    perturbation_strength=1.0
):
    """
    Evaluate a trained encoder checkpoint.
    
    Example:
        python train_encoder.py evaluate --checkpoint='checkpoints/encoder_epoch_50.pt'
    """
    device = torch.device(device)
    
    # Load model
    vq_model = VQ_models[vq_model](
        codebook_size=codebook_size,
        codebook_embed_dim=codebook_embed_dim
    )
    vq_model.to(device)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint, map_location=device)
    vq_model.encoder.load_state_dict(ckpt['encoder_state_dict'])
    vq_model.eval()
    
    print(f"Loaded checkpoint from epoch {ckpt['epoch']+1}")
    print(f"Training metrics - Loss: {ckpt['train_loss']:.4f}, Acc: {ckpt['train_accuracy']:.4f}")
    if 'perturbation_strength' in ckpt:
        print(f"Trained with perturbation strength: {ckpt['perturbation_strength']:.3f}")
    
    # Setup data
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Test with and without perturbations
    for use_perturb in ([False, True] if test_perturbations else [False]):
        if use_perturb:
            perturbations = ImagePerturbations(
                gaussian_noise_std=0.1,
                salt_pepper_prob=0.05,
                jpeg_quality_range=(30, 95),
                blur_kernel_size=11,
                apply_prob=1.0,
                strength_multiplier=perturbation_strength
            )
            print(f"\nEvaluating WITH perturbations (strength={perturbation_strength}):")
        else:
            perturbations = None
            print(f"\nEvaluating WITHOUT perturbations:")
        
        dataset = ImageTokenDataset(data_dir, transform=transform,
                                  perturbation_transform=perturbations)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Evaluate
        total_acc = 0
        
        with torch.no_grad():
            for images, original_tokens, _ in tqdm(loader, desc='Evaluating'):
                images = images.to(device)
                original_tokens = original_tokens.to(device)
                
                encoder_output = vq_model.encoder(images)
                _, indices, _ = vq_model.quantize(encoder_output)
                indices = indices.view(indices.shape[0], -1)
                
                accuracy = (indices == original_tokens).float().mean()
                total_acc += accuracy.item()
        
        print(f"Accuracy: {total_acc/len(loader):.4f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        # No arguments provided → call default
        train()
    else:
        # Expose all functions to Fire
        fire.Fire({
            'train': train,
            'visualize_schedule': visualize_schedule,
            'test_dataloader': test_dataloader,
            'evaluate': evaluate
        })