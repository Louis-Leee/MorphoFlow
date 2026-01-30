"""
Stage 2 training: Object-conditioned grasp generation with palm velocity head.

Loads RR weights from Stage 1, trains OR weights + palm velocity head from scratch.
Uses lower learning rate for RR weights (transferred from Stage 1).

Usage:
    python train_fm_stage2.py train.stage1_ckpt=graph_exp/fm_stage1/ckpt/epoch=199.ckpt
    python train_fm_stage2.py train.gpus=4
"""

import os
import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import hydra
from omegaconf import DictConfig, OmegaConf

torch.multiprocessing.set_sharing_strategy("file_system")

from model.flow_matching_twostage import TwoStageFlowMatchingGraph
from dataset.PalmCentricDataset import (
    PalmCentricStage2Dataset,
    palm_centric_collate_fn,
)


# Keywords that identify RR-related parameters (transferred from Stage 1)
RR_PARAM_KEYWORDS = [
    "V_robot_in_layers",
    "E_rr_in_layers",
    "v_robot_binding_fc",
    "e_rr_binding_fc",
    "rr_query_fc",
    "rr_key_fc",
    "rr_value_fc",
    "rr_r_self",
    "rr_out",
    "rr_edge_out",
    "v_robot_wide_fc",
    "v_robot_output_module",
    "link_token_encoder",
    "t_embed",  # sinusoidal embedding (shared)
]


def is_rr_param(name: str) -> bool:
    """Check if a parameter name belongs to RR-related weights."""
    for keyword in RR_PARAM_KEYWORDS:
        if keyword in name:
            return True
    return False


class Stage2Module(pl.LightningModule):
    """Lightning module for Stage 2 training with differential LR."""

    def __init__(self, model_cfg: dict, train_cfg):
        super().__init__()
        self.save_hyperparameters()
        self.model = TwoStageFlowMatchingGraph(**model_cfg)
        self.train_cfg = train_cfg

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        batch = self._prepare_input(batch)
        loss_dict = self.model(batch)
        bs = batch["object_pc"].shape[0]
        for k, v in loss_dict.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=(k == "loss_total"),
                sync_dist=True,
                batch_size=bs,
            )
        return loss_dict["loss_total"]

    def configure_optimizers(self):
        rr_lr_scale = self.train_cfg.fine_tune.get("rr_lr_scale", 0.1)
        base_lr = self.train_cfg.lr

        # Split parameters into RR (low LR) and non-RR (normal LR) groups
        rr_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if is_rr_param(name):
                rr_params.append(param)
            else:
                other_params.append(param)

        print(f"\nOptimizer parameter groups:")
        print(f"  RR params (lr={base_lr * rr_lr_scale:.1e}): {len(rr_params)} tensors")
        print(f"  Other params (lr={base_lr:.1e}): {len(other_params)} tensors")

        param_groups = [
            {"params": rr_params, "lr": base_lr * rr_lr_scale},
            {"params": other_params, "lr": base_lr},
        ]

        optimizer = torch.optim.Adam(param_groups)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.train_cfg.lr_step,
            gamma=self.train_cfg.lr_gamma,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def _prepare_input(self, batch):
        """Move batch tensors to the current device."""
        device = self.device
        batch["object_pc"] = batch["object_pc"].to(device)
        if "object_pc_normal" in batch:
            batch["object_pc_normal"] = batch["object_pc_normal"].to(device)

        for key in [
            "palm_centric_vec",
            "palm_centric_se3",
            "palm_pose_in_object",
            "initial_palm_centric_vec",
            "initial_palm_pose_in_object",
            "initial_q",
            "target_q",
            "initial_se3",
        ]:
            if key in batch and isinstance(batch[key], list):
                batch[key] = [
                    x.to(device) if torch.is_tensor(x) else x for x in batch[key]
                ]

        return batch


class Stage2DataModule(pl.LightningDataModule):
    def __init__(self, dataset_cfg):
        super().__init__()
        self.dataset_cfg = dataset_cfg

    def train_dataloader(self):
        dataset = PalmCentricStage2Dataset(
            batch_size=self.dataset_cfg.batch_size,
            robot_names=list(self.dataset_cfg.robot_names),
            is_train=True,
            debug_object_names=self.dataset_cfg.get("debug_object_names"),
            object_pc_type=self.dataset_cfg.get("object_pc_type", "random"),
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            collate_fn=palm_centric_collate_fn,
            num_workers=self.dataset_cfg.get("num_workers", 8),
            shuffle=True,
            persistent_workers=self.dataset_cfg.get("num_workers", 8) > 0,
        )


def load_stage1_weights(module, ckpt_path):
    """Load Stage 1 checkpoint weights into Stage 2 model."""
    print(f"\nLoading Stage 1 weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Handle Lightning checkpoint format
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Remove 'model.' prefix if present (Lightning wraps in module)
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("model.model."):
            cleaned[k[len("model.") :]] = v
        elif k.startswith("model."):
            cleaned[k] = v
        else:
            cleaned["model." + k] = v

    missing, unexpected = module.load_state_dict(cleaned, strict=False)

    print(f"  Loaded: {len(cleaned) - len(unexpected)} parameters")
    print(f"  Missing (new in Stage 2): {len(missing)}")
    if missing:
        for m in missing[:10]:
            print(f"    - {m}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")
    print(f"  Unexpected (Stage 1 only): {len(unexpected)}")

    # Count RR vs OR weights loaded
    rr_loaded = sum(1 for k in cleaned if is_rr_param(k) and k not in unexpected)
    or_loaded = sum(1 for k in cleaned if not is_rr_param(k) and k not in unexpected)
    print(f"  RR weights loaded: {rr_loaded}")
    print(f"  Other weights loaded: {or_loaded}")


@hydra.main(config_path="config", config_name="train_fm_stage2", version_base=None)
def main(cfg: DictConfig):
    print("\n" + "=" * 60)
    print("STAGE 2: Object-Conditioned Grasp Generation")
    print("=" * 60)
    print(f"Experiment: {cfg.train.experiment_name}")
    print(f"Robot Names: {list(cfg.dataset.robot_names)}")
    print(f"Batch Size: {cfg.dataset.batch_size}")
    print(f"RR LR Scale: {cfg.train.fine_tune.get('rr_lr_scale', 0.1)}")
    print(f"GPUs: {cfg.train.get('gpus', 1)}")
    print(f"Stage 1 Checkpoint: {cfg.train.get('stage1_ckpt', 'None')}")
    print(f"Save Dir: {cfg.train.save_dir}")
    print("=" * 60 + "\n")

    pl.seed_everything(cfg.get("seed", 42), workers=True)

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    train_cfg = cfg.train

    module = Stage2Module(model_cfg, train_cfg)

    # Load Stage 1 weights
    stage1_ckpt = cfg.train.get("stage1_ckpt")
    if stage1_ckpt:
        load_stage1_weights(module, stage1_ckpt)
    else:
        print("\nWARNING: No Stage 1 checkpoint provided. Training from scratch.")

    datamodule = Stage2DataModule(cfg.dataset)

    save_dir = cfg.train.save_dir
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=os.path.join(save_dir, "ckpt"),
        filename="{epoch}",
        save_top_k=-1,
        every_n_epochs=cfg.train.save_interval,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    logger = WandbLogger(
        project=cfg.train.project_name,
        name=cfg.train.experiment_name,
        save_dir=save_dir,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    num_gpus = cfg.train.get("gpus", 1)
    strategy = (
        DDPStrategy(find_unused_parameters=False) if num_gpus > 1 else "auto"
    )

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="gpu",
        devices=num_gpus,
        strategy=strategy,
        logger=logger,
        callbacks=[checkpoint_cb, lr_monitor],
        gradient_clip_val=cfg.train.get("gradient_clip_val", None),
        log_every_n_steps=1,
        precision=cfg.train.get("precision", 32),
    )

    ckpt_path = cfg.train.get("resume_from") or None
    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
