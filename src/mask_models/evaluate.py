"""Evaluate semantic-region adherence and generated-video health."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None
from .preprocessing import SegFormerMaskConfig, load_segformer, predict_id_maps
from src.data.frame_sampling import select_frame_indices


def read_video(path):
    if cv2 is None: raise RuntimeError("video evaluation requires OpenCV")
    cap=cv2.VideoCapture(str(path)); frames=[]
    if not cap.isOpened(): raise RuntimeError(f"Could not open {path}")
    while True:
        ok,frame=cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames: raise RuntimeError(f"No frames in {path}")
    return np.stack(frames)


def semantic_metrics(reference, generated, num_classes=150):
    if reference.shape != generated.shape: raise ValueError("semantic maps must match")
    present=np.union1d(np.unique(reference),np.unique(generated)); per_class={}
    for cid in present:
        r=reference==cid; g=generated==cid; union=np.logical_or(r,g).sum()
        per_class[str(int(cid))]=float(np.logical_and(r,g).sum()/union) if union else None
    values=[x for x in per_class.values() if x is not None]
    return {"pixel_agreement":float((reference==generated).mean()),"macro_iou_present":float(np.mean(values)) if values else None,"per_class_iou":per_class}


def boundary_maps(ids):
    edge=np.zeros_like(ids,dtype=bool); edge[:,:,1:] |= ids[:,:,1:] != ids[:,:,:-1]; edge[:,1:,:] |= ids[:,1:,:] != ids[:,:-1,:]
    return edge


def boundary_f1(reference, generated, tolerance=3):
    if cv2 is None: raise RuntimeError("boundary evaluation requires OpenCV")
    scores=[]; kernel=np.ones((2*tolerance+1,2*tolerance+1),np.uint8)
    for r,g in zip(boundary_maps(reference),boundary_maps(generated)):
        if not r.any() or not g.any(): scores.append(1.0 if not r.any() and not g.any() else 0.0); continue
        rn=cv2.dilate(r.astype(np.uint8),kernel)>0; gn=cv2.dilate(g.astype(np.uint8),kernel)>0
        p=(g&rn).sum()/g.sum(); q=(r&gn).sum()/r.sum(); scores.append(float(2*p*q/max(p+q,1e-12)))
    return float(np.mean(scores))


def boundary_chamfer(reference, generated):
    if cv2 is None: raise RuntimeError("boundary evaluation requires OpenCV")
    scores=[]
    for r,g in zip(boundary_maps(reference),boundary_maps(generated)):
        if not r.any() or not g.any():
            scores.append(None); continue
        dr=cv2.distanceTransform((~r).astype(np.uint8),cv2.DIST_L2,3)
        dg=cv2.distanceTransform((~g).astype(np.uint8),cv2.DIST_L2,3)
        scores.append(0.5*(float(dr[g].mean())+float(dg[r].mean())))
    values=[x for x in scores if x is not None]
    return float(np.mean(values)) if values else None


def semantic_temporal_agreement(ids):
    return float((ids[1:]==ids[:-1]).mean()) if len(ids)>1 else 1.0


def video_health(frames):
    x=frames.astype(np.float32); return {"mean":float(x.mean()),"std":float(x.std()),"temporal_mad":float(np.abs(x[1:]-x[:-1]).mean()) if len(x)>1 else 0.0}


def evaluate(reference_frames, base_frames, controlled_frames, extractor, processor, config, device):
    count=min(map(len,(reference_frames,base_frames,controlled_frames)))
    reference_frames=reference_frames[
        select_frame_indices(0, len(reference_frames), count)
    ]
    base_frames=base_frames[:count]; controlled_frames=controlled_frames[:count]
    runtime=SegFormerMaskConfig(count,config.output_size,config.batch_size)
    ref=predict_id_maps(reference_frames,processor,extractor,runtime,device=device); base=predict_id_maps(base_frames,processor,extractor,runtime,device=device); ctrl=predict_id_maps(controlled_frames,processor,extractor,runtime,device=device)
    frame0={"base":semantic_metrics(ref[:1],base[:1]),"controlled":semantic_metrics(ref[:1],ctrl[:1])}
    rest=slice(1,None) if count>1 else slice(0,None)
    effect=np.abs(base_frames.astype(np.float32)-controlled_frames.astype(np.float32)).mean(axis=(1,2,3))
    return {"frames":count,"matched_evaluator":True,"frame_zero":frame0,"frames_1_plus":{"base":semantic_metrics(ref[rest],base[rest]),"controlled":semantic_metrics(ref[rest],ctrl[rest]),"base_boundary_f1":boundary_f1(ref[rest],base[rest]),"controlled_boundary_f1":boundary_f1(ref[rest],ctrl[rest]),"base_boundary_chamfer":boundary_chamfer(ref[rest],base[rest]),"controlled_boundary_chamfer":boundary_chamfer(ref[rest],ctrl[rest])},"semantic_temporal_agreement":{"reference":semantic_temporal_agreement(ref),"base":semantic_temporal_agreement(base),"controlled":semantic_temporal_agreement(ctrl)},"class_diagnostics":{"reference_class_count":int(np.unique(ref).size),"base_class_count":int(np.unique(base).size),"controlled_class_count":int(np.unique(ctrl).size)},"video_health":{"base":video_health(base_frames),"controlled":video_health(controlled_frames)},"base_controlled_pixel_mad":{"all_frames":float(effect.mean()),"frames_1_plus":float(effect[1:].mean()) if count>1 else None}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ("reference","base","controlled","output"): p.add_argument(f"--{flag}",required=True)
    p.add_argument("--device",default="cuda:0"); p.add_argument("--cache-dir"); p.add_argument("--allow-download",action="store_true"); p.add_argument("--height",type=int,default=128); p.add_argument("--width",type=int,default=128); p.add_argument("--batch-size",type=int,default=4)
    a=p.parse_args(); config=SegFormerMaskConfig(output_size=(a.height,a.width),batch_size=a.batch_size); processor,model=load_segformer(config,cache_dir=a.cache_dir,local_files_only=not a.allow_download)
    report=evaluate(read_video(a.reference),read_video(a.base),read_video(a.controlled),model,processor,config,a.device)
    path=Path(a.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8"); print(json.dumps(report,indent=2))

if __name__=="__main__": main()
