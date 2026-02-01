"""
Flow Matching V3 training script with PyTorch Lightning + Hydra.

Uses FlowMatchingV3 (Flow Matching + FlashAttentionDenoiserNoEdge).
No-edge ablation: removes explicit SE(3) edge computation.
Direct linear interpolation for both translation and rotation.

Usage:
    # Single GPU
    python train_fm_v3.py

    # Multi-GPU DDP
    python train_fm_v3.py train.gpus=4

    # Override config via Hydra CLI
    python train_fm_v3.py train.lr=2e-4 dataset.batch_size=32

    # Resume from checkpoint
    python train_fm_v3.py train.resume_from=graph_exp/fm_v3/ckpt/epoch=49.ckpt
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

torch.multiprocessing.set_sharing_strategy('file_system')

from model.flow_matching_v3 import FlowMatchingV3
from dataset.CMapDataset import CMapDataset, custom_collate_fn


class FlowMatchingV3Module(pl.LightningModule):
    """Lightning module wrapping FlowMatchingV3 for DDP training."""

    def __init__(self, model_cfg: dict, train_cfg):
        super().__init__()
        self.save_hyperparameters()
        self.model = FlowMatchingV3(**model_cfg)
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
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_cfg.lr,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.train_cfg.lr_step,
            gamma=self.train_cfg.lr_gamma,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def _prepare_input(self, batch):
        """Move batch tensors to the current device."""
        device = self.device

        if "robot_pc_initial" in batch:
            for robot_pc in batch["robot_pc_initial"]:
                for link_name in robot_pc:
                    robot_pc[link_name] = robot_pc[link_name].to(device)

        if "robot_pc_target" in batch:
            for robot_pc in batch["robot_pc_target"]:
                for link_name in robot_pc:
                    robot_pc[link_name] = robot_pc[link_name].to(device)

        batch["object_pc"] = batch["object_pc"].to(device)
        if "object_pc_normal" in batch:
            batch["object_pc_normal"] = batch["object_pc_normal"].to(device)

        for key in [
            "initial_q",
            "target_q",
            "initial_se3",
            "target_se3",
            "initial_vec",
            "target_vec",
        ]:
            if key in batch and isinstance(batch[key], list):
                batch[key] = [x.to(device) for x in batch[key]]

        return batch


class GraspDataModule(pl.LightningDataModule):
    """DataModule wrapping CMapDataset for DDP-compatible training."""

    def __init__(self, dataset_cfg):
        super().__init__()
        self.dataset_cfg = dataset_cfg

    def train_dataloader(self):
        dataset = CMapDataset(
            batch_size=self.dataset_cfg.batch_size,
            robot_names=list(self.dataset_cfg.robot_names),
            is_train=True,
            debug_object_names=self.dataset_cfg.debug_object_names,
            object_pc_type=self.dataset_cfg.object_pc_type,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            collate_fn=custom_collate_fn,
            num_workers=self.dataset_cfg.num_workers,
            shuffle=True,
            persistent_workers=self.dataset_cfg.num_workers > 0,
        )


@hydra.main(config_path="config", config_name="train_fm_v3", version_base=None)
def main(cfg: DictConfig):
    print("\n" + "=" * 60)
    print("FLOW MATCHING V3 TRAINING (No Edge)")
    print("=" * 60)
    print(f"Experiment: {cfg.train.experiment_name}")
    print(f"Robot Names: {list(cfg.dataset.robot_names)}")
    print(f"Batch Size: {cfg.dataset.batch_size}")
    print(f"GPUs: {cfg.train.get('gpus', 1)}")
    print(f"Save Dir: {cfg.train.save_dir}")
    print("=" * 60 + "\n")

    pl.seed_everything(cfg.get("seed", 42), workers=True)

    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    train_cfg = cfg.train

    module = FlowMatchingV3Module(model_cfg, train_cfg)

    # Optional: load pretrained weights for transfer learning
    pretrained_path = cfg.train.get("pretrained_from")
    if pretrained_path:
        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
        missing, unexpected = module.model.load_state_dict(state_dict, strict=False)
        print(
            f"Loaded pretrained weights from {pretrained_path}. "
            f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )

    datamodule = GraspDataModule(cfg.dataset)

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
