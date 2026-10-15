"""Prepare deterministic raw SegFormer-B5 semantic-mask controls."""

from __future__ import annotations

import argparse, hashlib, json, subprocess
from pathlib import Path
import numpy as np

from src.data.frame_sampling import resolve_frame_interval, select_frame_indices
from .labels import ADE20K_LABEL_ORDER_SHA256, ADE20K_VISUALIZATION_PALETTE_SHA256, validate_numpy_id_map
from .preprocessing import SegFormerMaskConfig, load_segformer, predict_id_maps, processor_metadata, resolved_weights_sha256
from .temporal_stability import adjacent_agreement

FORMAT_VERSION = 1


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _output_path(root: Path, record: dict) -> Path:
    return root / str(record["video_id"]) / f"shot_{record['shot_id']}_controls_encoded.npz"


def read_selected_rgb(video: Path, record: dict, count: int):
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")
    actual = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start, end = resolve_frame_interval(int(record["segment_start_frame"]), int(record["segment_end_frame"]), actual)
    indices = select_frame_indices(start, end, count)
    frames = []
    try:
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f"Could not decode frame {index} from {video}")
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return np.stack(frames), [int(x) for x in indices]


def tensor_record(mask: np.ndarray, path: Path, root: Path, record: dict, indices: list[int], resumed=False):
    validate_numpy_id_map(mask, expected_ndim=5)
    flat = mask.reshape(-1)
    histogram = np.bincount(flat, minlength=150)
    maps = mask[0, 0]
    return {
        "video_id": str(record["video_id"]), "shot_id": str(record["shot_id"]),
        "source_video": f"{record['video_id']}.mp4",
        "path": str(path.relative_to(root)), "shape": list(mask.shape), "dtype": str(mask.dtype),
        "frame_indices": indices, "tensor_sha256": _sha(np.ascontiguousarray(mask).tobytes()),
        "class_histogram": histogram.tolist(), "class_count": int(np.count_nonzero(histogram)),
        "dominant_class": int(histogram.argmax()), "dominant_fraction": float(histogram.max()/flat.size),
        "temporal_agreement": float(adjacent_agreement(maps).mean()) if len(maps) > 1 else 1.0, "resumed": resumed,
    }


def prepare_record(record, videos_dir: Path, output_dir: Path, config, processor, model, *, device: str, resume=False, overwrite=False):
    video = videos_dir / f"{record['video_id']}.mp4"
    if not video.exists():
        raise FileNotFoundError(video)
    path = _output_path(output_dir, record)
    frames, indices = read_selected_rgb(video, record, config.num_frames)
    if path.exists() and resume:
        with np.load(path, allow_pickle=False) as data:
            if data.files != ["mask_encoded"]:
                raise ValueError(f"Invalid keys in {path}: {data.files}")
            mask = np.asarray(data["mask_encoded"])
        expected = (1, 1, config.num_frames, *config.output_size)
        if mask.dtype != np.uint8 or mask.shape != expected:
            raise ValueError(f"Invalid resumed tensor {path}: {mask.dtype} {mask.shape}")
        return tensor_record(mask, path, output_dir, record, indices, True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    ids = predict_id_maps(frames, processor, model, config, device=device)
    mask = ids[None, None]
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, mask_encoded=mask)
    temp.replace(path)
    return tensor_record(mask, path, output_dir, record, indices)


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ("videos-dir","metadata","output-dir"): p.add_argument(f"--{flag}", required=True)
    p.add_argument("--num-frames",type=int,default=8); p.add_argument("--height",type=int,default=128); p.add_argument("--width",type=int,default=128)
    p.add_argument("--batch-size",type=int,default=4); p.add_argument("--device",default="cuda:0"); p.add_argument("--cache-dir"); p.add_argument("--allow-download",action="store_true")
    g=p.add_mutually_exclusive_group(); g.add_argument("--resume",action="store_true"); g.add_argument("--overwrite",action="store_true")
    return p


def main():
    from tqdm import tqdm
    a=build_parser().parse_args(); videos=Path(a.videos_dir).resolve(); metadata=Path(a.metadata).resolve(); out=Path(a.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    config=SegFormerMaskConfig(a.num_frames,(a.height,a.width),a.batch_size)
    processor,model=load_segformer(config,cache_dir=a.cache_dir,local_files_only=not a.allow_download)
    records=sorted(json.loads(metadata.read_text(encoding="utf-8")),key=lambda x:str(x["shot_id"]))
    results=[prepare_record(r,videos,out,config,processor,model,device=a.device,resume=a.resume,overwrite=a.overwrite) for r in tqdm(records,desc="Preparing semantic masks")]
    try: commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    except Exception: commit=None
    manifest={"format_version":FORMAT_VERSION,"control_key":"mask_encoded","project_commit":commit,"metadata_path":str(metadata),"metadata_sha256":_sha(metadata.read_bytes()),"preprocessing":config.to_metadata(),"weights_sha256":resolved_weights_sha256(config,cache_dir=a.cache_dir,local_files_only=not a.allow_download),"processor":processor_metadata(processor),"label_order_sha256":ADE20K_LABEL_ORDER_SHA256,"visualization_palette_sha256":ADE20K_VISUALIZATION_PALETTE_SHA256,"records":results}
    target=out/"mask_preprocessing_manifest.json"; temp=target.with_suffix(".tmp.json"); temp.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8"); temp.replace(target)
    print(f"Prepared {len(results)} mask controls\nManifest: {target}")


if __name__ == "__main__": main()
