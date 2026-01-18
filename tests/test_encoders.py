

import torch
import sys
from pathlib import Path
import gc


def clear_gpu_memory():
    """Force clear all GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU Memory before clearing: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
      
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        
        print(f"GPU Memory after clearing: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"GPU Memory available: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1e9:.2f} GB\n")


clear_gpu_memory()


sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.encoders import DepthEncoder, SketchEncoder, MotionEncoder, StyleEncoder, MaskEncoder


class EncoderTester:
  
    
    def __init__(self, device='cuda'):
        self.device = device
        self.test_results = {}
    
    def test_encoder(
        self, 
        encoder_class, 
        input_shape, 
        expected_output_shape,
        test_backward=True
    ):
       
        print(f"\n{'='*60}")
        print(f"Testing {encoder_class.__name__}")
        print(f"{'='*60}")
        
       
        gc.collect()
        torch.cuda.empty_cache()
        
        try:
           
            encoder = encoder_class(out_channels=256).to(self.device)
            encoder.eval()  
            
           
            num_params = sum(p.numel() for p in encoder.parameters())
            trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
            
            print(f"✓ Initialization successful")
            print(f"  Total parameters: {num_params:,} ({num_params/1e6:.2f}M)")
            print(f"  Trainable parameters: {trainable_params:,}")
            
          
            print(f"\n  Creating test input...")
            x = torch.randn(*input_shape, device=self.device, dtype=torch.float32)
            print(f"✓ Input created: {x.shape}")
            print(f"  Memory after input: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            
           
            torch.cuda.reset_peak_memory_stats()
            
            with torch.no_grad():  
                y = encoder(x)
            
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            
            print(f"✓ Forward pass successful")
            print(f"  Output shape: {y.shape}")
            print(f"  Expected shape: {expected_output_shape}")
            print(f"  Peak memory: {peak_memory:.2f} GB")
            print(f"  Current memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            
            
            assert y.shape == expected_output_shape, \
                f"Shape mismatch! Got {y.shape}, expected {expected_output_shape}"
            print(f"✓ Output shape validated")
            
          
            assert not torch.isnan(y).any(), "Output contains NaN!"
            assert not torch.isinf(y).any(), "Output contains Inf!"
            print(f"✓ No NaN/Inf in outputs")
            
            
            self.test_results[encoder_class.__name__] = {
                'status': 'PASSED',
                'params': num_params,
                'memory_gb': peak_memory,
                'output_shape': y.shape
            }
            
            print(f"\n✅ {encoder_class.__name__} PASSED ALL TESTS")
            
          
            del encoder, x, y
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\n❌ {encoder_class.__name__} FAILED")
            print(f"Error: {str(e)}")
            self.test_results[encoder_class.__name__] = {
                'status': 'FAILED',
                'error': str(e)
            }
            
          
            gc.collect()
            torch.cuda.empty_cache()
            raise
    
    def run_all_tests(self):
        """Run tests for all encoders with REDUCED resolution."""
        B, T, H, W = 1, 16, 256, 256  
        H_out, W_out = H // 2, W // 2
        
        print("\n" + "="*60)
        print("ENCODER UNIT TESTS (256x256 resolution)")
        print("="*60)
        print(f"Using reduced resolution for memory efficiency")
        print(f"Input: [{B}, C, {T}, {H}, {W}]")
        print(f"Output: [{B}, 256, {T}, {H_out}, {W_out}]")
        
     
        self.test_encoder(
            DepthEncoder,
            input_shape=(B, 1, T, H, W),
            expected_output_shape=(B, 256, T, H_out, W_out),
            test_backward=False  
        )
        
      
        self.test_encoder(
            SketchEncoder,
            input_shape=(B, 1, T, H, W),
            expected_output_shape=(B, 256, T, H_out, W_out),
            test_backward=False
        )
        
        
        self.test_encoder(
            MotionEncoder,
            input_shape=(B, 2, T, H, W),
            expected_output_shape=(B, 256, T, H_out, W_out),
            test_backward=False
        )
        
    
        print(f"\n{'='*60}")
        print(f"Testing StyleEncoder")
        print(f"{'='*60}")
        
        gc.collect()
        torch.cuda.empty_cache()
        
        style_enc = StyleEncoder(out_channels=256).to(self.device)
        style_enc.eval()
        
        ref_img = torch.randn(B, 3, H, W, device=self.device)
        
        with torch.no_grad():
            style_out = style_enc(ref_img, num_frames=T)
        
        expected_style_shape = (B, 256, T, H // 8, W // 8)
        assert style_out.shape == expected_style_shape, \
            f"Style encoder shape mismatch! Got {style_out.shape}, expected {expected_style_shape}"
        print(f"✓ StyleEncoder passed: {style_out.shape}")
        
        self.test_results['StyleEncoder'] = {
            'status': 'PASSED',
            'params': sum(p.numel() for p in style_enc.parameters()),
            'memory_gb': torch.cuda.max_memory_allocated() / 1e9,
            'output_shape': style_out.shape
        }
        
        del style_enc, ref_img, style_out
        gc.collect()
        torch.cuda.empty_cache()
        
        
        self.test_encoder(
            MaskEncoder,
            input_shape=(B, 1, T, H, W),
            expected_output_shape=(B, 256, T, H_out, W_out),
            test_backward=False
        )
        
      
        self.print_summary()
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results.values() if r['status'] == 'PASSED')
        
        print(f"\nTotal Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        
        print("\nDetailed Results:")
        for name, result in self.test_results.items():
            status_icon = "✅" if result['status'] == 'PASSED' else "❌"
            print(f"{status_icon} {name}: {result['status']}")
            if result['status'] == 'PASSED':
                print(f"   Parameters: {result['params']/1e6:.2f}M")
                print(f"   Memory: {result['memory_gb']:.2f} GB")
        
        print("\n" + "="*60)
        print(f"Final GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print("="*60)


if __name__ == '__main__':
    
    print("Checking for existing GPU processes...")
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print(result.stdout)
    except:
        pass
    
 
    
    input("Press Enter to continue with tests...")
    
    tester = EncoderTester(device='cuda')
    tester.run_all_tests()