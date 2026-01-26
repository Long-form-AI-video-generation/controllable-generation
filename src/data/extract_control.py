import json
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import warnings
from collections import defaultdict
import gc
from PIL import Image

import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel
import mediapipe as mp
import urllib.request
import ssl
import os

warnings.filterwarnings('ignore')

class EnhancedControlExtractor:
    """
    Extract ALL control signals using existing VideoComposer models
    """
    def __init__(self, device='cuda', models_dir='../../models'):
        self.device = device
        self.models_dir = Path(models_dir)
        
        print("Loading models from VideoComposer weights...")
        
       
        self.midas_model, self.midas_transform = self.load_midas_local()
        
       
        self.clip_model, self.clip_processor = self.load_clip_local()
        
      
        self.pidinet_model = self.load_pidinet()
        
      
        self.pose_detector = self.load_pose_detector()
        
     
        self.face_detector = self.load_face_detector()
        
        
        
     
        self.normals_model = self.load_normals_model()
        
        print("✓ All models loaded!\n")
    
    
    def load_midas_local(self):
        """Load YOUR local MiDaS weights"""
        print("  Loading MiDaS (local)...")
        
        midas_path = self.models_dir / "midas_v3_dpt_large.pth"
        
        if not midas_path.exists():
            raise FileNotFoundError(f"MiDaS weights not found at {midas_path}")
        
        try:
          
            model_type = "DPT_Large"
            midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=False)
            
         
            checkpoint = torch.load(midas_path, map_location=self.device)
            midas.load_state_dict(checkpoint)
            
            midas.to(self.device)
            midas.eval()
            
         
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            transform = midas_transforms.dpt_transform
            
            print(f"    ✓ MiDaS loaded from {midas_path.name}")
            return midas, transform
            
        except Exception as e:
            print(f"    ⚠️  Failed to load local MiDaS, downloading default...")
          
            midas = torch.hub.load("intel-isl/MiDaS", model_type)
            midas.to(self.device)
            midas.eval()
            
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            transform = midas_transforms.dpt_transform
            
            return midas, transform
    
    def load_clip_local(self):
        """Load YOUR local CLIP weights"""
        print("  Loading CLIP (local)...")
        
        clip_path = self.models_dir / "open_clip_pytorch_model.bin"
        
        try:
          
            from transformers import CLIPModel, CLIPProcessor
            
            model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            
         
            if clip_path.exists():
                print(f"    Loading weights from {clip_path.name}...")
                state_dict = torch.load(clip_path, map_location=self.device)
                
             
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
               
                try:
                    model.load_state_dict(state_dict, strict=False)
                    print(f"    ✓ Loaded local CLIP weights")
                except Exception as e:
                    print(f"    ⚠️  Using default CLIP weights: {e}")
            
            model.to(self.device)
            model.eval()
            
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            
            return model, processor
            
        except Exception as e:
            print(f"    ❌ CLIP loading failed: {e}")
            raise
    
    def load_pidinet(self):
        """Load YOUR PiDiNet model for better edge detection"""
        print("  Loading PiDiNet (local)...")
        
        pidinet_path = self.models_dir / "table5_pidinet.pth"
        
        if not pidinet_path.exists():
            print("    ⚠️  PiDiNet not found, will use Canny fallback")
            return None
        
        try:
            
            print("    ⚠️  PiDiNet requires model definition, using Canny")
            return None
            
        except Exception as e:
            print(f"    ⚠️  PiDiNet loading failed: {e}")
            return None
    
    def load_pose_detector(self):
        """Load pose detector"""
        print("  Loading Pose Detector...")
        
      
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8n-pose.pt')
            print("    ✓ Using YOLOv8-Pose")
            return {'type': 'yolo', 'model': model}
        except:
            pass
        
      
        try:
            pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5
            )
            print("    ✓ Using MediaPipe Pose")
            return {'type': 'mediapipe', 'model': pose}
        except Exception as e:
            print(f"    ⚠️  No pose detection available")
            return None
    
    def load_face_detector(self):
        """Load face detector"""
        print("  Loading Face Detector...")
        
      
        anime_cascade = Path('models/lbpcascade_animeface.xml')
        if anime_cascade.exists():
            detector = cv2.CascadeClassifier(str(anime_cascade))
            if not detector.empty():
                print("    ✓ Using anime face detector")
                return detector
        
     
        detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        print("    ✓ Using default face detector")
        return detector
    
    def load_normals_model(self):
        """Load surface normals model (Omnidata DPT)"""
        print("  Loading Surface Normals Model...")
        
        normals_path = self.models_dir / "omnidata_dpt_normal_v2.ckpt"
        model_url = "https://huggingface.co/clay3d/omnidata/resolve/main/omnidata_dpt_normal_v2.ckpt"
        
        # 1. Download if not exists
        if not normals_path.exists():
            print(f"    ⚠️  Omnidata model not found locally.")
            print(f"    Downloading from {model_url}...")
            print(f"    (This is a ~2GB file, please wait...)")
            
            try:
                # Create unverified context to avoid SSL errors
                ssl_context = ssl._create_unverified_context()
                
                with urllib.request.urlopen(model_url, context=ssl_context) as response:
                    with open(normals_path, 'wb') as out_file:
                        # Simple download loop to show some progress could be added,
                        # but for now we just read/write
                        block_size = 1024 * 1024  # 1MB
                        while True:
                            buffer = response.read(block_size)
                            if not buffer:
                                break
                            out_file.write(buffer)
                
                print(f"    ✓ Download complete: {normals_path.name}")
                
            except Exception as e:
                print(f"    ❌ Download failed: {e}")
                print("    Using fallback: Derive normals from depth")
                return None

        # 2. Load Model
        try:
            print("    Loading DPT architecture...")
            # We use the DPT_Large architecture from MiDaS as the backbone
            model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", pretrained=False)
            model.to(self.device)
            model.eval()
            
            print(f"    Loading weights from {normals_path.name}...")
            checkpoint = torch.load(normals_path, map_location=self.device)
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load weights (strict=False because of potential minor key differences)
            model.load_state_dict(state_dict, strict=False)
            
            # Setup transform
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.normals_transform = midas_transforms.dpt_transform
            
            print(f"    ✓ Loaded Omnidata Normals Model")
            return model
            
        except Exception as e:
            print(f"    ⚠️  Normals model loading failed: {e}")
            print("    Using fallback: Derive normals from depth")
            return None
    
    # === EXTRACTION METHODS ===
    
    def extract_depth(self, frame, target_size=(360, 640)):
        """Extract depth using YOUR MiDaS model"""
        input_batch = self.midas_transform(frame).to(self.device)
        with torch.no_grad():
            prediction = self.midas_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size,
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        depth_map = (depth_map * 255).astype(np.uint8)
        return depth_map
    
    def extract_edges(self, frame):
        """Extract edges - using Canny"""
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return edges
    
    def extract_optical_flow(self, prev_frame, curr_frame, flow_clip_range=30):
        """Extract optical flow"""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0
        )
        
        flow_clipped = np.clip(flow, -flow_clip_range, flow_clip_range)
        return flow_clipped.astype(np.float16)
    
    def extract_style_embedding(self, frame):
        """Extract style using YOUR CLIP model"""
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = self.clip_processor(images=pil_image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().astype(np.float16)
    
    def extract_pose(self, frame):
        """Extract pose keypoints"""
        if self.pose_detector is None:
            return np.zeros((17, 3), dtype=np.float16)
        
        try:
            if self.pose_detector['type'] == 'yolo':
                results = self.pose_detector['model'](frame, verbose=False)
                
                
                if len(results) > 0 and results[0].keypoints is not None:
                    keypoints_data = results[0].keypoints.data
                    
                   
                    if keypoints_data.shape[0] > 0:
                        kpts = keypoints_data[0].cpu().numpy()
                        return kpts.astype(np.float16)
                
              
                return np.zeros((17, 3), dtype=np.float16)
            
            elif self.pose_detector['type'] == 'mediapipe':
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose_detector['model'].process(frame_rgb)
                
                if results.pose_landmarks:
                    h, w = frame.shape[:2]
                    landmarks = []
                    for landmark in results.pose_landmarks.landmark:
                        landmarks.append([
                            landmark.x * w,
                            landmark.y * h,
                            landmark.visibility
                        ])
                    return np.array(landmarks, dtype=np.float16)
                
                return np.zeros((33, 3), dtype=np.float16)  
            
        except Exception as e:
            
            return np.zeros((17, 3), dtype=np.float16)
        
        return np.zeros((17, 3), dtype=np.float16)
    
    def extract_face_info(self, frame):
        """Detect faces and extract embeddings"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24)
        )
        
        face_data = {
            'boxes': [],
            'embeddings': []
        }
        
        for (x, y, w, h) in faces:
            face_data['boxes'].append((x, y, w, h))
            
            if w > 0 and h > 0:
                face_crop = frame[y:y+h, x:x+w]
                try:
                    face_emb = self.extract_style_embedding(face_crop)
                    face_data['embeddings'].append(face_emb)
                except:
                    face_data['embeddings'].append(None)
        
        return face_data
    
    def extract_color_palette(self, frame, n_colors=8):
        """Extract dominant colors"""
        pixels = frame.reshape(-1, 3).astype(np.float32)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        return centers.astype(np.uint8)
    
    def extract_lighting(self, frame):
        """Estimate lighting conditions"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        brightness = np.mean(gray) / 255.0
        contrast = np.std(gray) / 255.0
        
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        
        return {
            'brightness': np.float16(brightness),
            'contrast': np.float16(contrast),
            'light_direction': np.array([np.mean(grad_x), np.mean(grad_y)], dtype=np.float16)
        }
    
    def estimate_camera_motion(self, prev_frame, curr_frame):
        """Estimate camera motion"""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(curr_gray, None)
        
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return {
                'translation': np.zeros(2, dtype=np.float16),
                'rotation': np.float16(0),
                'scale': np.float16(1.0)
            }
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        if len(matches) < 4:
            return {
                'translation': np.zeros(2, dtype=np.float16),
                'rotation': np.float16(0),
                'scale': np.float16(1.0)
            }
        
        matches = sorted(matches, key=lambda x: x.distance)[:50]
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        M, _ = cv2.estimateAffinePartial2D(pts1, pts2)
        
        if M is None:
            return {
                'translation': np.zeros(2, dtype=np.float16),
                'rotation': np.float16(0),
                'scale': np.float16(1.0)
            }
        
        translation = M[:, 2]
        scale = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
        rotation = np.arctan2(M[1, 0], M[0, 0])
        
        return {
            'translation': translation.astype(np.float16),
            'rotation': np.float16(rotation),
            'scale': np.float16(scale)
        }
    

    
    def extract_surface_normals(self, frame, depth_map=None, target_size=(360, 640)):
        """
        Extract surface normals using Omnidata model or derive from depth.
        """
        # 1. Use Omnidata model if available
        if self.normals_model is not None:
            try:
                # Transform input
                input_batch = self.normals_transform(frame).to(self.device)
                
                with torch.no_grad():
                    prediction = self.normals_model(input_batch)
                    
                    # Resize to target
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=target_size,
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()
                    
                    
                    if len(prediction.shape) == 3 and prediction.shape[0] == 3:
                        # [3, H, W] -> [H, W, 3]
                        prediction = prediction.permute(1, 2, 0)
                    
                    normal_map = prediction.cpu().numpy()
                    
                    # Normalize to [0, 255]
                    # Usually output is [-1, 1]
                    normal_map = ((normal_map + 1.0) * 127.5)
                    normal_map = np.clip(normal_map, 0, 255).astype(np.uint8)
                    
                    return normal_map
                    
            except Exception as e:
                print(f"    ⚠️  Omnidata extraction failed: {e}")
                # Fallthrough to depth-based fallback
        
        # 2. Fallback: derive from depth
        if depth_map is None:
            depth_map = self.extract_depth(frame, target_size)
        
        # Ensure depth is float and normalized
        if depth_map.dtype == np.uint8:
            depth_map = depth_map.astype(np.float32) / 255.0
        
        # Compute gradients (surface derivatives)
        zy, zx = np.gradient(depth_map.astype(np.float32))
        
        # Compute normal vectors: (-dz/dx, -dz/dy, 1)
        # The negative signs account for depth convention
        normal = np.dstack((-zx, -zy, np.ones_like(depth_map)))
        
        # Normalize to unit length
        n = np.linalg.norm(normal, axis=2, keepdims=True)
        normal = normal / (n + 1e-8)
        
        # Convert from [-1, 1] to [0, 255]
        normal_map = ((normal + 1.0) * 127.5).astype(np.uint8)
        
        return normal_map
    
    


def process_shot_with_all_controls(
    video_path,
    shot,
    output_dir,
    extractor: EnhancedControlExtractor,
    sample_rate=4,
    target_size=(360, 640),
    extract_full_controls=True
):
    """Extract ALL control signals from a shot"""
    video_id = shot['video_id']
    shot_idx = shot['shot_id']
    start_frame = shot['segment_start_frame']
    end_frame = shot['segment_end_frame']
    num_frames = end_frame - start_frame
    
    if num_frames < sample_rate:
        return False
    
    output_path = Path(output_dir) / video_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / f"shot_{shot_idx}_controls.npz"
    if output_file.exists():
        return True
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    
    try:
        frames_to_process = list(range(0, num_frames, sample_rate))
        
        controls = {
            'depth': [],
            'edges': [],
            'flow': [],
            'reference_frame': None,
            'style_embedding': None,
            'pose_sequence': [],
            'color_palette_sequence': [],
            'lighting_sequence': [],
            'camera_motion': [],
            'face_detections': [],
            'normals': [],
        }
        
        prev_frame = None
        first_frame_saved = False
        
        pbar = tqdm(
            frames_to_process,
            desc=f"  Shot {shot_idx}",
            leave=False,
            unit="frame"
        )
        
        for frame_offset in frames_to_process:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + frame_offset)
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_resized = cv2.resize(frame, target_size[::-1])
            
            # Core extractions
            depth = extractor.extract_depth(frame_resized, target_size)
            controls['depth'].append(depth)
            
            edges = extractor.extract_edges(frame_resized)
            controls['edges'].append(edges)
          
            normals = extractor.extract_surface_normals(frame_resized, depth_map=depth, target_size=target_size)
            controls['normals'].append(normals)
            
            
            
            if prev_frame is not None:
                flow = extractor.extract_optical_flow(prev_frame, frame_resized)
                controls['flow'].append(flow)
            
            # Extended extractions
            if extract_full_controls:
                if not first_frame_saved:
                    controls['reference_frame'] = frame_resized
                    controls['style_embedding'] = extractor.extract_style_embedding(frame_resized)
                    first_frame_saved = True
                
                pose = extractor.extract_pose(frame_resized)
                controls['pose_sequence'].append(pose)
                
                palette = extractor.extract_color_palette(frame_resized)
                controls['color_palette_sequence'].append(palette)
                
                lighting = extractor.extract_lighting(frame_resized)
                controls['lighting_sequence'].append(lighting)
                
                face_info = extractor.extract_face_info(frame_resized)
                controls['face_detections'].append(face_info)
                
                if prev_frame is not None:
                    camera = extractor.estimate_camera_motion(prev_frame, frame_resized)
                    controls['camera_motion'].append(camera)
            
            prev_frame = frame_resized.copy()
            pbar.update(1)
            
            
            if len(controls['depth']) % 50 == 0:
                torch.cuda.empty_cache()
        
        pbar.close()
        cap.release()
        
       
        if len(controls['depth']) == 0:
            return False
        
      
        save_data = {
            'depth': np.stack(controls['depth']),
            'edges': np.stack(controls['edges']),
            'normals': np.stack(controls['normals']),
            'metadata': {
                'video_id': video_id,
                'shot_id': shot_idx,
                'sample_rate': sample_rate,
                'target_size': target_size,
                'full_controls_extracted': extract_full_controls,
            }
        }
        
        if controls['flow']:
            save_data['flow'] = np.stack(controls['flow'])
        
        if extract_full_controls:
            save_data['reference_frame'] = controls['reference_frame']
            save_data['style_embedding'] = controls['style_embedding']
            
            if controls['pose_sequence']:
                save_data['pose_sequence'] = np.stack(controls['pose_sequence'])
            if controls['color_palette_sequence']:
                save_data['color_palettes'] = np.stack(controls['color_palette_sequence'])
            if controls['lighting_sequence']:
                save_data['lighting'] = controls['lighting_sequence']
            if controls['camera_motion']:
                save_data['camera_motion'] = controls['camera_motion']
            if controls['face_detections']:
                save_data['face_detections'] = controls['face_detections']
        
        np.savez_compressed(output_file, **save_data)
        
        return True
        
    except Exception as e:
        print(f"\n  ❌ Error processing shot {shot_idx}: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cap.release()
        torch.cuda.empty_cache()
        gc.collect()


def process_dataset(
    video_dir,
    shot_dir,
    output_dir,
    device='cuda',
    sample_rate=4,
    target_size=(360, 640),
    extract_full_controls=True,
    extractor=None
):
    """Process entire dataset"""
    print(f"\n{'='*70}")
    print(f"Enhanced Control Signal Extraction")
    print(f"{'='*70}\n")
    
    # Load metadata
    with open(shot_dir, 'r') as f:
        all_shots = json.load(f)
    
    shots_by_video = defaultdict(list)
    for shot in all_shots:
        shots_by_video[shot['video_id']].append(shot)
    
    print(f"Loaded {len(all_shots)} shots from {len(shots_by_video)} videos\n")
    
    
    if extractor is None:
        extractor = EnhancedControlExtractor(device=device)
    
    
    total_processed = 0
    total_failed = 0
    
    video_pbar = tqdm(shots_by_video.items(), desc="Videos", unit="video")
    
    for video_id, shots in video_pbar:
        video_path = Path(video_dir) / f"{video_id}.mp4"
        if not video_path.exists():
            total_failed += len(shots)
            continue
        
        for shot in shots:
            success = process_shot_with_all_controls(
                video_path=video_path,
                shot=shot,
                output_dir=output_dir,
                extractor=extractor,
                sample_rate=sample_rate,
                target_size=target_size,
                extract_full_controls=extract_full_controls
            )
            
            if success:
                total_processed += 1
            else:
                total_failed += 1
        
        video_pbar.set_postfix({
            'Processed': total_processed,
            'Failed': total_failed
        })
    
    video_pbar.close()
    
    print(f"\n{'='*70}")
    print(f"Complete! Processed: {total_processed}, Failed: {total_failed}")
    print(f"{'='*70}\n")

def main():
    SHOTS_METADATA = "../../data/shots_metadata.json"
    VIDEOS_DIR = "../../data/videos"
    OUTPUT_DIR = "../../data/control_signals"
    MODELS_DIR = "../../models"  # YOUR models
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using models from: {MODELS_DIR}\n")
    
  
    extractor = EnhancedControlExtractor(device=DEVICE, models_dir=MODELS_DIR)
    
  
    process_dataset(
        video_dir=VIDEOS_DIR,
        shot_dir=SHOTS_METADATA,
        output_dir=OUTPUT_DIR,
        device=DEVICE,
        sample_rate=4,
        target_size=(360, 640),
        extract_full_controls=True
    )


if __name__ == "__main__":
    main()