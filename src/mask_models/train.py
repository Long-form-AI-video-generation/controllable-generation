"""Train the mask-only WAN control adapter.

This entry point deliberately has no machine-specific paths. All data, WAN, and
checkpoint locations are supplied on the command line by the execution host.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import ControllableVideoDataset
from src.depth_models.train import (
    MultiVideoTrainer,
    flow_matching_loss,
    temporal_smoothness_loss,
    timestep_weighted_flow_loss,
)
from src.mask_models.wan_controllable import ControllableWAN
from src.mask_models.training_schedule import expected_optimizer_steps


CONTROL_KEY = "mask_encoded"


def _captions_to_device(captions, device: str):
    if isinstance(captions, torch.Tensor):
        return captions.to(device)
    return list(captions)


class MaskTrainer(MultiVideoTrainer):
    """Trainer specialized for one-channel semantic mask control."""

    def _loss_step(self, batch, *, training: bool) -> tuple[torch.Tensor, dict]:
        video = batch["video"].to(self.device)
        controls = {
            CONTROL_KEY: batch["controls"][CONTROL_KEY].to(self.device)
        }
        prompts = _captions_to_device(batch["caption"], self.device)

        with torch.no_grad():
            latent = self.base_model.encode_video(video)
        del video

        batch_size = latent.shape[0]
        if training:
            timesteps = torch.randint(
                50, 950, (batch_size,), device=self.device
            ).long()
            noise = torch.randn_like(latent, dtype=torch.float32)
        else:
            generator = self._validation_generator
            timesteps = torch.randint(
                0,
                1000,
                (batch_size,),
                device=self.device,
                generator=generator,
            ).long()
            noise = torch.randn(
                latent.shape,
                dtype=torch.float32,
                device=latent.device,
                generator=generator,
            )

        interpolation = (
            timesteps.float() / 1000.0
        ).view(-1, 1, 1, 1, 1)
        noisy = (1.0 - interpolation) * latent + interpolation * noise
        target = noise - latent
        del interpolation, noise, latent

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=self.config["mixed_precision"],
        ):
            prediction = self.base_model(
                latent=noisy,
                timesteps=timesteps,
                prompts=prompts,
                control_features=controls,
            )

        flow = flow_matching_loss(prediction, target)
        weighted = timestep_weighted_flow_loss(
            prediction, target, timesteps
        )
        temporal = temporal_smoothness_loss(prediction)
        total = (
            self.config["loss_flow_weight"] * flow
            + self.config["loss_weighted_weight"] * weighted
            + self.config["loss_temporal_weight"] * temporal
        )

        metrics = {
            "loss": float(total.detach()),
            "loss_flow": float(flow.detach()),
            "loss_weighted": float(weighted.detach()),
            "loss_temporal": float(temporal.detach()),
            "loss_temporal_weighted": float(
                self.config["loss_temporal_weight"] * temporal.detach()
            ),
            "lr_adapter": self.optimizer.param_groups[0]["lr"],
            "lr_zero_conv": self.optimizer.param_groups[1]["lr"],
            "timestep_mean": float(timesteps.float().mean()),
        }
        return total, metrics

    def train_step(self, batch) -> tuple[torch.Tensor, dict]:
        return self._loss_step(batch, training=True)

    @torch.no_grad()
    def val_step(self, batch) -> dict:
        _, metrics = self._loss_step(batch, training=False)
        return metrics

    def _parameter_group_stats(self) -> dict[str, float]:
        statistics = {}
        for group in self.optimizer.param_groups:
            name = group.get("name", "unnamed")
            parameters = list(group["params"])
            parameter_square = sum(
                float(parameter.detach().float().norm()) ** 2
                for parameter in parameters
            )
            gradient_square = sum(
                float(parameter.grad.detach().float().norm()) ** 2
                for parameter in parameters
                if parameter.grad is not None
            )
            statistics[f"{name}_parameter_norm"] = parameter_square ** 0.5
            statistics[f"{name}_gradient_norm"] = gradient_square ** 0.5
        return statistics

    def _optimizer_step(self, accumulated_batches: int) -> dict[str, float]:
        target = self.config["grad_accum_steps"]
        self.scaler.unscale_(self.optimizer)
        if accumulated_batches < target:
            correction = target / accumulated_batches
            for parameter in self.base_model.get_trainable_parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
        statistics = self._parameter_group_stats()
        torch.nn.utils.clip_grad_norm_(
            self.base_model.get_trainable_parameters(),
            self.config["max_grad_norm"],
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.lr_scheduler.step()
        self.global_step += 1
        return statistics

    def _after_optimizer_step(self, epoch: int, metrics: dict) -> None:
        if self.global_step % self.config["log_every"] == 0:
            record = {
                "step": self.global_step,
                "epoch": epoch,
                **metrics,
                **self.get_zero_conv_stats(),
                **self.get_modality_gate_stats(),
                "gpu0_memory_gb": torch.cuda.memory_allocated(0) / 1e9,
            }
            if torch.cuda.device_count() > 1:
                record["gpu1_memory_gb"] = (
                    torch.cuda.memory_allocated(1) / 1e9
                )
            self.log_metrics(record)

        if self.global_step % self.config["save_every"] == 0:
            self.save_checkpoint(f"step_{self.global_step}", epoch=epoch)

        if self.global_step % self.config["val_every"] == 0:
            validation_loss = self.validate()
            if validation_loss < self.best_val_loss:
                self.best_val_loss = validation_loss
                self.save_checkpoint("best", epoch=epoch)
            self.model.train()

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses = []
        accumulated = 0
        progress = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")

        for batch_index, batch in enumerate(progress):
            try:
                loss, metrics = self.train_step(batch)
                scaled = loss / self.config["grad_accum_steps"]
                self.scaler.scale(scaled).backward()
                losses.append(metrics["loss"])
                accumulated += 1
                progress.set_postfix(
                    loss=f"{metrics['loss']:.4f}",
                    avg=f"{sum(losses) / len(losses):.4f}",
                    step=self.global_step,
                )

                boundary = accumulated == self.config["grad_accum_steps"]
                final_batch = batch_index + 1 == len(self.train_loader)
                if boundary or final_batch:
                    metrics.update(self._optimizer_step(accumulated))
                    accumulated = 0
                    self._after_optimizer_step(epoch, metrics)
                    if self.global_step >= self.config["num_steps"]:
                        break
            except torch.OutOfMemoryError:
                self.optimizer.zero_grad(set_to_none=True)
                accumulated = 0
                torch.cuda.empty_cache()
                gc.collect()
                raise RuntimeError(
                    f"CUDA OOM at epoch {epoch + 1}, batch {batch_index}; "
                    "strict mask training does not silently skip samples"
                )

        return sum(losses) / len(losses) if losses else float("inf")

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        self._validation_generator = torch.Generator(device=self.device)
        self._validation_generator.manual_seed(
            self.config["validation_seed"]
        )
        losses = []
        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            metrics = self.val_step(batch)
            losses.append(metrics["loss"])
        average = sum(losses) / len(losses) if losses else float("inf")
        self.log_metrics({"step": self.global_step, "val_loss": average})
        print(f"  Validation Loss: {average:.4f}")
        return average

    def _checkpoint_payload(self, epoch: int) -> dict:
        return {
            "model": self.base_model.control_adapter.state_dict(),
            "zero_convs": self.base_model.zero_convs.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "control_metadata": self.base_model.checkpoint_metadata(),
        }

    def save_checkpoint(self, name: str, epoch: int = 0) -> None:
        path = self.checkpoint_dir / f"checkpoint_{name}.pt"
        temporary = path.with_suffix(".tmp.pt")
        torch.save(self._checkpoint_payload(epoch), temporary)
        temporary.replace(path)
        print(f"  Checkpoint saved: {path}")

    def export_for_inference(self, name: str = "final") -> None:
        from safetensors.torch import save_file

        common = {
            "control_type": "mask",
            "control_key": CONTROL_KEY,
            "training_step": str(self.global_step),
            "cfg_control_policy": self.base_model.cfg_control_policy,
        }
        adapter_path = self.checkpoint_dir / f"mask_adapter_{name}.safetensors"
        zero_convs_path = (
            self.checkpoint_dir / f"mask_zero_convs_{name}.safetensors"
        )
        adapter_temporary = adapter_path.with_suffix(".tmp.safetensors")
        zero_convs_temporary = zero_convs_path.with_suffix(".tmp.safetensors")
        save_file(
            self.base_model.control_adapter.state_dict(),
            str(adapter_temporary),
            metadata={**common, "component": "adapter_and_condition_encoder"},
        )
        save_file(
            self.base_model.zero_convs.state_dict(),
            str(zero_convs_temporary),
            metadata={**common, "component": "zero_convs"},
        )
        adapter_temporary.replace(adapter_path)
        zero_convs_temporary.replace(zero_convs_path)
        print(f"  Exported: {adapter_path}")
        print(f"  Exported: {zero_convs_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        expected = self.base_model.checkpoint_metadata()
        actual = checkpoint.get("control_metadata")
        if actual != expected:
            raise ValueError(
                "Checkpoint is not compatible with this mask model. "
                f"Expected {expected}, got {actual}"
            )
        saved_config = checkpoint.get("config", {})
        compatibility_keys = (
            "control_key",
            "num_frames",
            "resolution",
            "split_manifest_sha256",
            "preprocessing_manifest_sha256",
        )
        mismatches = {
            key: (saved_config.get(key), self.config.get(key))
            for key in compatibility_keys
            if saved_config.get(key) != self.config.get(key)
        }
        if mismatches:
            raise ValueError(
                f"Checkpoint run configuration is incompatible: {mismatches}"
            )
        self.base_model.control_adapter.load_state_dict(checkpoint["model"])
        self.base_model.zero_convs.load_state_dict(checkpoint["zero_convs"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.scaler.load_state_dict(checkpoint["scaler"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.start_epoch = checkpoint.get("epoch", 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--wan-dir", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--preprocessing-manifest", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--num-steps", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=(128, 128),
        metavar=("HEIGHT", "WIDTH"),
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--validation-seed", type=int, default=20260808)
    parser.add_argument("--no-mixed-precision", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Mask training requires CUDA")
    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            "Current WAN wrapper requires two GPUs (DiT on 0, VAE on 1)"
        )

    split_manifest_path = Path(args.split_manifest).resolve()
    preprocessing_manifest_path = Path(
        args.preprocessing_manifest
    ).resolve()
    preprocessing_manifest = json.loads(
        preprocessing_manifest_path.read_text(encoding="utf-8")
    )
    if preprocessing_manifest.get("control_key") != CONTROL_KEY:
        raise ValueError(
            "Preprocessing manifest is not a mask_encoded dataset"
        )
    preprocessing = preprocessing_manifest.get("preprocessing", {})
    required_manifest_fields = (
        "weights_sha256", "label_order_sha256",
        "visualization_palette_sha256", "processor",
    )
    missing = [key for key in required_manifest_fields if key not in preprocessing_manifest]
    if missing:
        raise ValueError(f"Preprocessing manifest is missing {missing}")
    if preprocessing.get("num_frames") != args.num_frames:
        raise ValueError(
            "Preprocessing and training frame counts do not match"
        )
    if tuple(preprocessing.get("output_size", ())) != tuple(args.resolution):
        raise ValueError(
            "Preprocessing and training control resolutions do not match"
        )

    config = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "num_epochs": args.epochs,
        "num_steps": args.num_steps,
        "grad_accum_steps": args.grad_accum_steps,
        "mixed_precision": not args.no_mixed_precision,
        "num_frames": args.num_frames,
        "resolution": tuple(args.resolution),
        "lr": args.lr,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "loss_flow_weight": 1.0,
        "loss_weighted_weight": 0.1,
        "loss_temporal_weight": 0.05,
        "log_every": 10,
        "save_every": args.save_every,
        "val_every": args.val_every,
        "validation_seed": args.validation_seed,
        "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
        "data_dir": str(Path(args.data_dir).resolve()),
        "checkpoint_path": str(Path(args.wan_dir).resolve()),
        "control_key": CONTROL_KEY,
        "split_manifest": str(split_manifest_path),
        "split_manifest_sha256": hashlib.sha256(
            split_manifest_path.read_bytes()
        ).hexdigest(),
        "preprocessing_manifest": str(preprocessing_manifest_path),
        "preprocessing_manifest_sha256": hashlib.sha256(
            preprocessing_manifest_path.read_bytes()
        ).hexdigest(),
        "preprocessing": {
            **preprocessing,
            "weights_sha256": preprocessing_manifest["weights_sha256"],
            "processor_sha256": preprocessing_manifest["processor"]["sha256"],
        },
        "text_preprocessing": {
            "encoder": "WAN 2.2 UMT5-XXL",
            "sequence_length": 512,
            "padding": "zero embedding rows",
        },
    }

    model = ControllableWAN(
        checkpoint_dir=config["checkpoint_path"],
        device="cuda",
    )
    unexpected_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith(("control_adapter.", "zero_convs."))
    ]
    if unexpected_trainable:
        raise RuntimeError(
            f"Unexpected trainable WAN parameters: {unexpected_trainable[:10]}"
        )
    dataset_arguments = {
        "encoded_controls_dir": f"{config['data_dir']}/encoded_controls",
        "videos_dir": f"{config['data_dir']}/videos",
        "annotations_path": f"{config['data_dir']}/shots_metadata.json",
        "num_frames": config["num_frames"],
        "resolution": (
            config["resolution"][1],
            config["resolution"][0],
        ),
        "text_encoder": model,
        "load_videos": True,
        "control_key": CONTROL_KEY,
        "strict": True,
        "split_manifest_path": str(split_manifest_path),
    }
    train_dataset = ControllableVideoDataset(
        split="train", **dataset_arguments
    )
    val_dataset = ControllableVideoDataset(
        split="val", **dataset_arguments
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=False,
    )
    trainer = MaskTrainer(
        model,
        train_loader,
        val_loader,
        config,
        device="cuda",
        resume_from=args.resume,
    )
    scheduled_steps = expected_optimizer_steps(
        len(train_loader),
        config["grad_accum_steps"],
        config["num_epochs"],
    )
    if scheduled_steps != config["num_steps"]:
        raise ValueError(
            f"Configured run produces {scheduled_steps} optimizer steps, "
            f"but --num-steps is {config['num_steps']}"
        )

    active_epoch = trainer.start_epoch
    try:
        for epoch in range(trainer.start_epoch, config["num_epochs"]):
            active_epoch = epoch
            average = trainer.train_epoch(epoch)
            print(
                f"Epoch {epoch + 1}/{config['num_epochs']}: "
                f"loss={average:.4f}, step={trainer.global_step}"
            )
            if (epoch + 1) % 5 == 0:
                trainer.save_checkpoint(f"epoch_{epoch + 1}", epoch + 1)
                trainer.export_for_inference(f"epoch_{epoch + 1}")
            if trainer.global_step >= config["num_steps"]:
                break
    except KeyboardInterrupt:
        trainer.save_checkpoint("interrupted", active_epoch)
        trainer.export_for_inference("interrupted")
        print("Training interrupted; resumable checkpoint saved")
        return
    except Exception:
        trainer.save_checkpoint("error", active_epoch)
        raise

    final_validation = trainer.validate()
    if final_validation < trainer.best_val_loss:
        trainer.best_val_loss = final_validation
        trainer.save_checkpoint("best", config["num_epochs"])
    trainer.save_checkpoint("final", config["num_epochs"])
    trainer.export_for_inference("final")
    print(json.dumps({
        "final_val_loss": final_validation,
        "best_val_loss": trainer.best_val_loss,
        "global_step": trainer.global_step,
    }, indent=2))


if __name__ == "__main__":
    main()
