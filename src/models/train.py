import torch
from torch.utils.data import DataLoader
from models.control_adapter import ControlAdapter
from models.encoders import *  # Your encoders
from data.controllable_dataset import ControllableVideoDataset

def train():
    
    device = 'cuda'
    
   
    encoders = {
        'depth': DepthEncoder(out_channels=256).to(device).eval(),
        'sketch': SketchEncoder(out_channels=256).to(device).eval(),
        'motion': MotionEncoder(out_channels=256),
        'style': StyleEncoder(out_channels=256),
        'pose': PoseEncoder(out_channels=256),
        'mask': MaskEncoder(out_channels=256)
    }
    for enc in encoders.values():
        for param in enc.parameters():
            param.requires_grad = False
    
    
    control_adapter = ControlAdapter(
        control_dim=256,
        hidden_dim=1024,
        dit_dim=2048,
        num_controls=6
    ).to(device)
    
  
    from peft import LoraConfig, get_peft_model
    wan_model = load_wan_model()  
    
    lora_config = LoraConfig(
        r=8,  
        lora_alpha=16,
        target_modules=["qkv", "proj"], 
        lora_dropout=0.1
    )
    wan_model = get_peft_model(wan_model, lora_config)
    
    dataset = ControllableVideoDataset(
        encoded_controls_dir='data/encoded_controls',
        videos_dir='data/videos',
        annotations_path='data/annotations.json',
        num_frames=8
    )
    

    batch_size = 1 
    grad_accum_steps = 8
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
   
    trainable_params = list(control_adapter.parameters()) + \
                      list(wan_model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    
    
    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            controls = {k: v.to(device) for k, v in batch['controls'].items()}
            frames = batch['frames'].to(device)
            caption = batch['caption']
          
            control_signal = control_adapter(controls)
         
            loss = wan_model(
                frames,
                caption,
                control_signal=control_signal
            )
         
            loss = loss / grad_accum_steps
            loss.backward()
            
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()