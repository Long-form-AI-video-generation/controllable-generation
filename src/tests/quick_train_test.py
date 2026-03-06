
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from single_video_dataset import SingleVideoDataset
from models.control_adapter import ControlAdapter


def quick_train_test():
    
    
    print("="*70)
    print("Quick Training Test")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
  
    print("\n[1/4] Loading dataset...")
    dataset = SingleVideoDataset(
        encoded_controls_dir='data/test_single_video/encoded_controls',
        videos_dir='data/videos',
        annotations_path='data/test_single_video/shots_metadata.json',
        split='train',
        load_videos=False  
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    print(f"   ✓ Dataset size: {len(dataset)}")
    
    print("\n[2/4] Creating control adapter...")
    adapter = ControlAdapter(
        control_dim=256,
        hidden_dim=1024,
        dit_dim=2048,
        num_controls=6,  
        use_gradient_checkpointing=False
    ).to(device)
    
    print(f"   ✓ Adapter parameters: {sum(p.numel() for p in adapter.parameters()):,}")
    
   
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4)
   
    print("\n[3/4] Testing forward pass...")
    
    batch = next(iter(dataloader))
    controls = {k: v.to(device) for k, v in batch['controls'].items()}
    
    print(f"   Controls loaded: {list(controls.keys())}")
    
    with torch.cuda.amp.autocast():
        control_signal = adapter(controls)
    
    print(f"   ✓ Output shape: {control_signal.shape}")
    print(f"   ✓ Output range: [{control_signal.min():.3f}, {control_signal.max():.3f}]")
    
    print("\n[4/4] Testing backward pass...")
    
    target = torch.randn_like(control_signal)
    loss = F.mse_loss(control_signal, target)
    
    loss.backward()
    optimizer.step()
    
    print(f"   ✓ Loss: {loss.item():.4f}")
    print(f"   ✓ Gradients computed successfully")
    
    print("\n[5/5] Testing 5 training steps...")
    
    for step in range(5):
        batch = next(iter(dataloader))
        controls = {k: v.to(device) for k, v in batch['controls'].items()}
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            control_signal = adapter(controls)
            target = torch.randn_like(control_signal)
            loss = F.mse_loss(control_signal, target)
        
        loss.backward()
        optimizer.step()
        
        print(f"   Step {step+1}/5: loss = {loss.item():.4f}")
    
    print("\n" + "="*70)
    print("✅ Quick Training Test Complete!")
    print("="*70)
    print("\n Pipeline is ready for full training!")
    print("\nNext steps:")
    print("  1. Integrate with WAN model")
    print("  2. Add VAE for video encoding")
    print("  3. Add text encoder (CLIP)")
    print("  4. Run full training loop")


if __name__ == '__main__':
    quick_train_test()