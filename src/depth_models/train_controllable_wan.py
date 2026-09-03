import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from pathlib import Path
import sys
from tqdm import tqdm
import json
import gc

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'Wan2.2'))

from data.dataset import ControllableVideoDataset
from models.wan_controllable import ControllableWAN


class MultiVideoTrainer:
    
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config: dict,
        device: str = 'cuda'
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
      
        if torch.cuda.device_count() > 1:
            print(f"\n Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(model)
            self.base_model = model
        else:
            print("Only 1 GPU detected, running on single GPU")
            self.model = model
            self.base_model = model
        
        
        trainable_params = self.base_model.get_trainable_parameters()
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_steps'],
            eta_min=config['lr'] * 0.1
        )
        
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            prediction_type="epsilon"
        )
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=config['mixed_precision'])
        
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.checkpoint_dir / 'training_log.jsonl'
        
        print(f"\n{'='*70}")
        print("Multi-Video Trainer Initialized")
        print(f"{'='*70}")
        print(f"  GPUs: {torch.cuda.device_count()}")
        print(f"  Optimizer: AdamW (lr={config['lr']})")
        print(f"  Mixed Precision: {config['mixed_precision']}")
        print(f"  Gradient Accumulation: {config['grad_accum_steps']} steps")
        print(f"  Effective Batch Size: {config['batch_size'] * config['grad_accum_steps']}")
        print(f"  Trainable Params: {sum(p.numel() for p in trainable_params):,}")
        print(f"{'='*70}\n")
    
    def train_step(self, batch):
        
        video = batch['video'].to(self.device)
        controls = {k: v.to(self.device) for k, v in batch['controls'].items()}
        caption = batch['caption']
        print("caption", caption)
        with torch.no_grad():
            latent = self.base_model.encode_video(video)
        
        del video
        torch.cuda.empty_cache()
        
        batch_size = latent.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device
        ).long()
        
        noise = torch.randn_like(latent, dtype=torch.float32)
        noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)
        del latent
        
        noise_pred = self.base_model(
            latent=noisy_latent,
            timesteps=timesteps,
            prompts=caption,
            control_features=controls
        )
        
        loss = F.mse_loss(noise_pred, noise, reduction='mean')
        
        del noisy_latent, noise, noise_pred, controls
        torch.cuda.empty_cache()
        
        return loss, {
            'loss': loss.item(),
            'lr': self.optimizer.param_groups[0]['lr'],
            'timestep_mean': timesteps.float().mean().item()
        }
    
    def train_epoch(self, epoch):
        
        self.model.train()
        
        epoch_losses = []
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            try:
                loss, metrics = self.train_step(batch)
                
                loss = loss / self.config['grad_accum_steps']
                self.scaler.scale(loss).backward()
                
                if (step + 1) % self.config['grad_accum_steps'] == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.base_model.get_trainable_parameters(),
                        self.config['max_grad_norm']
                    )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    self.lr_scheduler.step()
                    self.global_step += 1
                    torch.cuda.empty_cache()
                    if self.global_step % self.config['log_every'] == 0:
                        self.log_metrics({
                            'step': self.global_step,
                            'epoch': epoch,
                            'train_loss': metrics['loss'],
                            'lr': metrics['lr'],
                            'gpu0_memory_gb': torch.cuda.memory_allocated(0)/1e9,
                            'gpu1_memory_gb': torch.cuda.memory_allocated(1)/1e9 if torch.cuda.device_count() > 1 else 0
                        })
                
                
                epoch_losses.append(metrics['loss'])
                
                progress_bar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'gpu0': f"{torch.cuda.memory_allocated(0)/1e9:.1f}GB",
                    'gpu1': f"{torch.cuda.memory_allocated(1)/1e9:.1f}GB" if torch.cuda.device_count() > 1 else "N/A",
                    'step': self.global_step
                })
                
                
                if self.global_step % self.config['save_every'] == 0:
                    self.save_checkpoint(f'step_{self.global_step}')
                
                if self.global_step % self.config['val_every'] == 0:
                    val_loss = self.validate()
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint('best')
                        print(f"  New best validation loss: {val_loss:.4f}")
                    
                    self.model.train()
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n  OOM at step {step}, clearing cache and skipping batch...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e
        
        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """Validation - same as single video"""
        self.model.eval()
        val_losses = []
        
        for batch in self.val_loader:
            try:
                loss, metrics = self.train_step(batch)
                val_losses.append(metrics['loss'])
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n OOM during validation, skipping batch...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        avg_loss = sum(val_losses) / len(val_losses) if val_losses else float('inf')
        print(f"\n  Validation Loss: {avg_loss:.4f}")
        
        self.log_metrics({
            'step': self.global_step,
            'val_loss': avg_loss
        })
        
        return avg_loss
    
    def save_checkpoint(self, name: str):
        """Save checkpoint - same as single video"""
        checkpoint_path = self.checkpoint_dir / f'checkpoint_{name}.pt'
        
        checkpoint = {
            'model': self.base_model.control_adapter.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f" Checkpoint saved: {checkpoint_path}")
    
    def export_for_inference(self, name: str = 'final'):
        """Export model as safetensors"""
        from safetensors.torch import save_file
        
        output_path = self.checkpoint_dir / f'control_adapter_{name}.safetensors'
        state_dict = self.base_model.control_adapter.state_dict()
        
        metadata = {
            'format': 'pt',
            'model_type': 'ControlAdapter',
            'num_params': str(sum(p.numel() for p in self.base_model.control_adapter.parameters())),
            'training_step': str(self.global_step),
            'best_val_loss': str(self.best_val_loss)
        }
        
        save_file(state_dict, str(output_path), metadata=metadata)
        print(f" Model exported: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    
    def log_metrics(self, metrics: dict):
        """Log metrics to file"""
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')


def main():
    """Main training function - MULTI-VIDEO VERSION"""
    print("\n" + "="*70)
    print("Training Controllable WAN - MULTIPLE VIDEOS")
    print("="*70)
    
    # Check GPUs
    print(f"\n Detecting GPUs...")
    print(f"   Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    
    config = {
        'batch_size': 1,
        'num_workers': 0,
        
        'num_epochs': 50,  
        'num_steps': 50000, 
        'grad_accum_steps': 8,
        'mixed_precision': True,
        
        'num_frames': 4,
        'resolution': (64, 64), 
        'control_resolution': (16, 16),
        
        'lr': 1e-4,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        
        'log_every': 10,
        'save_every': 1000,  
        'val_every': 500,  
        'checkpoint_dir': 'checkpoints/multi_video',
        
        
        'data_dir': '/mnt/d1/controllable-generation',
        'checkpoint_path': 'Wan2.2/Wan2.2-TI2V-5B'
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    torch.cuda.empty_cache()
    gc.collect()
    
    
    print("\n[1/4] Loading FULL dataset...")
    full_dataset = ControllableVideoDataset(
        encoded_controls_dir=f"{config['data_dir']}/encoded_controls",  
        videos_dir=f"{config['data_dir']}/videos",
        annotations_path=f"{config['data_dir']}/shots_metadata.json",
        num_frames=config['num_frames'],
        resolution=config['resolution'],
        split='train',
        load_videos=True
    )
    
    
    val_dataset = ControllableVideoDataset(
        encoded_controls_dir=f"{config['data_dir']}/encoded_controls",
        videos_dir=f"{config['data_dir']}/videos",
        annotations_path=f"{config['data_dir']}/shots_metadata.json",
        num_frames=config['num_frames'],
        resolution=config['resolution'],
        split='val',
        load_videos=True
    )
    
    print(f"   Train: {len(full_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")
    
    train_loader = DataLoader(
        full_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=False
    )
    
    print("\n[2/4] Loading Controllable WAN...")
    model = ControllableWAN(
        checkpoint_dir=config['checkpoint_path'],
        device=device
    )
    
    print("\n[3/4] Creating trainer...")
    trainer = MultiVideoTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    print("\n[4/4] Starting training...")
    print(f"{'='*70}\n")
    
    try:
        for epoch in range(config['num_epochs']):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{config['num_epochs']}")
            print(f"{'='*70}")
            
            avg_loss = trainer.train_epoch(epoch)
            
            print(f"\n  Epoch {epoch+1} Summary:")
            print(f"    Average Loss: {avg_loss:.4f}")
            print(f"    Global Step: {trainer.global_step}")
            print(f"    GPU 0 Memory: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
            if torch.cuda.device_count() > 1:
                print(f"    GPU 1 Memory: {torch.cuda.memory_allocated(1)/1e9:.2f} GB")
            
            if (epoch + 1) % 5 == 0:
                trainer.save_checkpoint(f'epoch_{epoch+1}', epoch=epoch+1)
                trainer.export_for_inference(f'epoch_{epoch+1}')
                print(f"  Saved checkpoint at epoch {epoch+1}")
            if trainer.global_step >= config['num_steps']:
                print(f"\n  Reached max steps ({config['num_steps']})")
                break
                      

        
        print("\n" + "="*70)
        print("Final Validation")
        print("="*70)
        final_val_loss = trainer.validate()
        
        trainer.save_checkpoint('final')
        trainer.export_for_inference('final')
        
        print("\n" + "="*70)
        print(" Training Complete!")
        print("="*70)
        print(f"  Best Validation Loss: {trainer.best_val_loss:.4f}")
        print(f"  Final Validation Loss: {final_val_loss:.4f}")
        print(f"  Total Steps: {trainer.global_step}")
        print(f"  Checkpoints: {config['checkpoint_dir']}")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n Training interrupted by user")
        trainer.save_checkpoint('interrupted')
    
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        trainer.save_checkpoint('error')


if __name__ == '__main__':
    main()