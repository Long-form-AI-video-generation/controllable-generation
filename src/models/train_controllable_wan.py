
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import sys
from typing import Dict, Optional


WAN_PATH = Path(__file__).parent / 'Wan2.2'
sys.path.insert(0, str(WAN_PATH))

from wan.models.diffusion import DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from models.wan_controllable import load_controllable_wan
from data.dataset import ControllableVideoDataset


class ControllableWANTrainer:
    """
    Trainer for Controllable WAN 2.2
    Memory-optimized for 2x RTX 3090
    """
    
    def __init__(
        self,
       
        encoded_dir: str,
        annotations_path: str,
       
        wan_config: str,
        wan_checkpoint: str,
        vae_checkpoint: str,
      
        output_dir: str = 'checkpoints/controllable_wan',
        batch_size: int = 1,  
        grad_accum_steps: int = 8, 
        num_workers: int = 2,
        lr: float = 1e-4,
        num_epochs: int = 50,
       
        use_lora: bool = True,
        lora_rank: int = 8,
       
        device: str = 'cuda'
    ):
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.grad_accum_steps = grad_accum_steps
        self.num_epochs = num_epochs
        
        print(f"\n{'='*70}")
        print("Initializing Controllable WAN Trainer")
        print(f"{'='*70}")
        
        print("\n[1/6] Loading training dataset...")
        self.train_dataset = ControllableVideoDataset(
            encoded_controls_dir=encoded_dir,
            videos_dir='data/videos',  
            annotations_path=annotations_path,
            split='train',
            load_videos=True
        )
        
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
      
        print("\n[1b/6] Loading validation dataset...")
        self.val_dataset = ControllableVideoDataset(
            encoded_controls_dir=encoded_dir,
            videos_dir='data/videos',
            annotations_path=annotations_path,
            split='val',
            load_videos=True
        )
        
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        
        print("\n[2/5] Loading text encoder...")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = self.text_encoder.to(device).eval()
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
      
        print("\n[3/5] Loading Controllable WAN...")
        self.model = load_controllable_wan(
            wan_config=wan_config,
            wan_checkpoint=wan_checkpoint,
            vae_checkpoint=vae_checkpoint,
            use_lora=use_lora,
            lora_rank=lora_rank,
            device=device
        )
        
      
        print("\n[4/5] Setting up diffusion scheduler...")
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear"
        )
       
        print("\n[5/5] Setting up optimizer...")
        trainable_params = self.model.get_trainable_parameters()
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        total_steps = len(self.dataloader) * num_epochs // grad_accum_steps
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=lr * 0.1
        )
        
      
        self.scaler = GradScaler()
        
       
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        print(f"\n{'='*70}")
        print("Trainer Initialized Successfully")
        print(f"{'='*70}")
        print(f"  Dataset size: {len(self.dataset)}")
        print(f"  Batch size: {batch_size}")
        print(f"  Gradient accumulation: {grad_accum_steps}")
        print(f"  Effective batch size: {batch_size * grad_accum_steps}")
        print(f"  Steps per epoch: {len(self.dataloader) // grad_accum_steps}")
        print(f"  Total steps: {total_steps}")
        print(f"  Learning rate: {lr}")
        print(f"{'='*70}\n")
    
    @torch.no_grad()
    def encode_text(self, captions: list) -> torch.Tensor:
        """Encode text captions to embeddings"""
        
        inputs = self.tokenizer(
            captions,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        

        outputs = self.text_encoder(**inputs)
        embeddings = outputs.last_hidden_state
        
        return embeddings


    @torch.no_grad()
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        print("  Running validation...")
        for batch in tqdm(self.val_dataloader, desc="Validation", leave=False):
          
            controls = {k: v.to(self.device) for k, v in batch['controls'].items()}
            frames = batch['frames'].to(self.device)
            captions = batch['caption']
            
            text_embeddings = self.encode_text(captions)
            latent = self.model.encode_video(frames)
            
            B = latent.shape[0]
            timesteps = torch.randint(
                0, self.noise_scheduler.num_train_timesteps,
                (B,), device=self.device
            ).long()
            
            noise = torch.randn_like(latent)
            noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)
            
            with autocast():
                noise_pred = self.model(
                    noisy_latent,
                    timesteps,
                    text_embeddings,
                    control_features=controls
                )
                loss = F.mse_loss(noise_pred, noise)
            
            total_loss += loss.item()
            num_batches += 1
        
        self.model.train()
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def train_step(self, batch: Dict) -> float:
      
        controls = {k: v.to(self.device) for k, v in batch['controls'].items()}
        frames = batch['frames'].to(self.device)  
        captions = batch['caption']
        
     
        text_embeddings = self.encode_text(captions)
        
       
        with torch.no_grad():
            latent = self.model.encode_video(frames)  
        
      
        B = latent.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps,
            (B,), device=self.device
        ).long()
        
     
        noise = torch.randn_like(latent)
        noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)
        
      
        with autocast():
           
            noise_pred = self.model(
                noisy_latent,
                timesteps,
                text_embeddings,
                control_features=controls
            )
            
          
            loss = F.mse_loss(noise_pred, noise)
     
        self.scaler.scale(loss).backward()
        
        return loss.item()
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.dataloader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1
            
            
            if (batch_idx + 1) % self.grad_accum_steps == 0:
               
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.get_trainable_parameters(),
                    max_norm=1.0
                )
                
               
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
               
                self.scheduler.step()
                
                self.global_step += 1
           
            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'avg_loss': f'{total_loss/num_batches:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                'step': self.global_step
            })
            
           
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        return total_loss / num_batches
    
    def save_checkpoint(self, train_loss: float, val_loss: float):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_loss': self.best_loss
        }
        
        
        path = self.output_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, path)
        print(f"    ✓ Saved: {path}")
        
      
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            path = self.output_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, path)
            print(f"    ✓ New best: {path} (val_loss: {val_loss:.4f})")
        
        
        if self.epoch % 5 == 0:
            path = self.output_dir / f'checkpoint_epoch_{self.epoch}.pth'
            torch.save(checkpoint, path)
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*70}")
        print(f"Starting Training - {self.num_epochs} Epochs")
        print(f"{'='*70}\n")
        
        for epoch in range(1, self.num_epochs + 1):
            self.epoch = epoch
            
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            print("-" * 70)
            
            
            train_loss = self.train_epoch()
           
            val_loss = self.validate()
            
           
            print(f"\n  Saving checkpoint...")
            self.save_checkpoint(train_loss, val_loss)
            
            print(f"\n  Epoch {epoch} Summary:")
            print(f"    Train Loss: {train_loss:.4f}")
            print(f"    Val Loss: {val_loss:.4f}")
            print(f"    Best Val Loss: {self.best_loss:.4f}")
            print(f"    Global Step: {self.global_step}")
        
        print(f"\n{'='*70}")
        print("Training Complete!")
        print(f"{'='*70}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument('--encoded_dir', type=str,
                       default='data/encoded_controls')
    parser.add_argument('--annotations', type=str,
                       default='data/annotations.json')
  
    parser.add_argument('--wan_config', type=str,
                       default='Wan2.2/configs/wan2.2_config.yaml')
    parser.add_argument('--wan_checkpoint', type=str,
                       default='checkpoints/wan2.2')
    parser.add_argument('--vae_checkpoint', type=str,
                       default='checkpoints/vae')
    
    
    parser.add_argument('--output_dir', type=str,
                       default='checkpoints/controllable_wan')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lora_rank', type=int, default=8)
    
    args = parser.parse_args()
    
   
    trainer = ControllableWANTrainer(
        encoded_dir=args.encoded_dir,
        annotations_path=args.annotations,
        wan_config=args.wan_config,
        wan_checkpoint=args.wan_checkpoint,
        vae_checkpoint=args.vae_checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        num_epochs=args.epochs,
        lr=args.lr,
        lora_rank=args.lora_rank
    )
    
    
    trainer.train()


if __name__ == '__main__':
    main()
