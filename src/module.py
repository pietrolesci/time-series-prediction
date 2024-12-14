import math
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property

from lightning.pytorch import LightningModule
from torch import Tensor
from torch.nn.functional import cross_entropy, mse_loss
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection, R2Score

from src.models import MODEL_CONFIG
from src.utilities import DictConfig, get_logger

logger = get_logger("module")


@dataclass
class OptimCofig(DictConfig):
    # Optimizer config
    optim_name: str
    lr: float
    weight_decay: float = 0.0
    optim_kwargs: dict = field(default_factory=dict)

    # Scheduler config
    scheduler_name: str | None = None
    num_warmup_steps: int | None = None
    scheduler_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.optim_name in TYPE_TO_OPTIMIZER_CLASS
        if self.scheduler_name is not None:
            assert self.scheduler_name in TYPE_TO_SCHEDULER_FUNCTION


"""
Lightning Module
"""


class RunningStage(StrEnum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


def get_reg_metrics(stage: str | RunningStage, dl_name: str | None = None) -> MetricCollection:
    metrics = {"r2": R2Score()}
    return MetricCollection(metrics, prefix=f"{stage}/", postfix=f"_{dl_name}" if dl_name else "")  # type: ignore


# def get_clf_metrics(stage: str | RunningStage) -> MetricCollection:
#     kwargs = {"task": "multiclass", "num_classes": TOTAL_NUMBER_OF_CLASSES}
#     metrics = {
#         "f1_micro": F1Score(average="micro", **kwargs),
#         "f1_macro": F1Score(average="macro", **kwargs),
#         "precision": Precision(average="macro", **kwargs),
#         "recall": Recall(average="macro", **kwargs),
#         "accuracy": Accuracy(**kwargs),
#         "mse": MeanSquaredError(),
#     }
#     return MetricCollection(metrics, prefix=f"{stage}/")


class LOBModel(LightningModule):
    def __init__(self, config: MODEL_CONFIG, optim_config: OptimCofig) -> None:
        super().__init__()
        self.config = config  # save it here so that we can find it in the checkpoints!
        self.optim_config = optim_config
        self.save_hyperparameters()

        # Having the config instantiating the model allows us to not pass the model class explicitly
        self.model = config.get_model()
        self.loss_fn = cross_entropy if self.config.is_classification else mse_loss

        # Define metrics
        for stage in RunningStage:
            if stage == RunningStage.VALIDATION:
                for dl_name in [RunningStage.VALIDATION, RunningStage.TEST]:
                    setattr(self, f"{stage}_reg_metrics_{dl_name}", get_reg_metrics(stage, dl_name))
            else:
                setattr(self, f"{stage}_reg_metrics", get_reg_metrics(stage))
            # if self.config.is_classification:
            #     setattr(self, f"{stage}_clf_metrics", get_clf_metrics(stage))

    @cached_property
    def should_mask_padding(self) -> bool:
        return self.trainer.datamodule.should_mask_padding  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        return self.model.forward(x)

    def step(self, batch: dict[str, Tensor], stage: RunningStage, suffix: str = "") -> dict[str, Tensor]:
        # unpack the batch and forward pass
        x, y, idx = batch["X"], batch["y"], batch["idx"]
        preds = self.forward(x)

        # prepare outputs and logs
        detached_pred = preds.detach()
        out = {"preds": detached_pred, "y": y, "idx": idx}
        out = {k: v[..., -1] if v.dim() > 1 else v for k, v in out.items()}
        logs = {
            "pred_mean": detached_pred.mean(),
            "pred_std": detached_pred.std(),
            "pred_min": detached_pred.min(),
            "pred_max": detached_pred.max(),
        }
        log_kwargs = {
            "on_step": stage == RunningStage.TRAIN,
            "on_epoch": stage != RunningStage.TRAIN,
            "prog_bar": False,
            "logger": True,
            "batch_size": y.shape[0],
            "add_dataloader_idx": False,
        }

        # compute loss and log it
        if self.should_mask_padding:
            mask = y != -100
            preds, y = preds[mask], y[mask]  # these get flattened
        loss = self.loss_fn(preds, y)

        # log
        logs["loss"] = loss.detach()
        self.log_dict({f"{stage}/{k}{suffix}": v for k, v in logs.items()}, **log_kwargs)

        # compute and log metrics
        reg_metrics = getattr(self, f"{stage}_reg_metrics{suffix}")
        # if self.config.is_classification:
        #     # for regression metrics convert to continuous values
        #     reg_metrics(logits_to_continuous(preds), class_to_continuous(y))

        #     # for classification keep it as is
        #     clf_metrics = getattr(self, f"{stage}_clf_metrics{suffix}")
        #     clf_metrics(preds, y)
        #     self.log_dict(clf_metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        # else:
        reg_metrics(preds, y)
        self.log_dict(reg_metrics, **log_kwargs)

        return {"loss": loss, **out}

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Tensor]:
        return self.step(batch, RunningStage.TRAIN)

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int, dataloader_idx: int) -> dict[str, Tensor]:
        dl_name = RunningStage.VALIDATION if dataloader_idx == 0 else RunningStage.TEST
        return self.step(batch, RunningStage.VALIDATION, f"_{dl_name}")

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> dict[str, Tensor]:
        return self.step(batch, RunningStage.TEST)

    def configure_optimizers(self) -> dict:
        # Get params that require grad
        param_dict = {pn: p for pn, p in self.model.named_parameters() if p.requires_grad}

        # Create optim_groups
        decay_params, nodecay_params = [], []
        for _, p in param_dict.items():
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)
        optim_groups = [
            {"params": decay_params, "weight_decay": self.optim_config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        # Create optimizer
        opt = TYPE_TO_OPTIMIZER_CLASS[self.optim_config.optim_name](
            optim_groups, lr=self.optim_config.lr, **self.optim_config.optim_kwargs
        )
        out: dict = {"optimizer": opt}

        # Maybe create scheduler
        if self.optim_config.scheduler_name is not None and self.optim_config.num_warmup_steps is not None:
            lr_scheduler = TYPE_TO_SCHEDULER_FUNCTION[self.optim_config.scheduler_name](
                optimizer=opt,
                num_warmup_steps=self.optim_config.num_warmup_steps,
                num_training_steps=int(self.trainer.estimated_stepping_batches),
                **self.optim_config.scheduler_kwargs,
            )
            out["lr_scheduler"] = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        logger.info(f"{len(decay_params)=}, with {sum(p.numel() for p in decay_params):,} params")
        logger.info(f"{len(nodecay_params)=}, with {sum(p.numel() for p in nodecay_params):,} params")
        logger.info(f"Optimisation info: {out}")

        return out


"""
Optimisers and Schedulers
"""


def get_linear_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, last_epoch: int = -1
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, last_epoch: int = -1
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_wsd_schedule(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_stable_steps: int,
    num_decay_steps: int,
    min_lr_ratio: float = 0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a schedule with a learning rate that has three stages:
    1. linear increase from 0 to initial lr.
    2. constant lr (equal to initial lr).
    3. decrease following the values of the cosine function between the initial lr set in the optimizer to
       a fraction of initial lr.

    Args:
        num_stable_steps (`int`):
            The number of steps for the stable phase.
        num_decay_steps (`int`):
            The number of steps for the cosine annealing phase.
        min_lr_ratio (`float`, *optional*, defaults to 0):
            The minimum learning rate as a ratio of the initial learning rate.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step < num_warmup_steps + num_stable_steps:
            return 1.0
        if current_step < num_warmup_steps + num_stable_steps + num_decay_steps:
            progress = float(current_step - num_warmup_steps - num_stable_steps) / float(max(1, num_decay_steps))
            value = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
            return (1.0 - min_lr_ratio) * value + min_lr_ratio
        return min_lr_ratio

    return LambdaLR(optimizer, lr_lambda, last_epoch)


TYPE_TO_OPTIMIZER_CLASS = {"adamw": AdamW}

TYPE_TO_SCHEDULER_FUNCTION = {
    "linear_with_warmup": get_linear_schedule_with_warmup,
    "constant_with_warmup": get_constant_schedule_with_warmup,
    "cosine_with_warmup": get_cosine_schedule_with_warmup,
    "cosine_hard_restarts_with_warmup": get_cosine_with_hard_restarts_schedule_with_warmup,
    "wsd_schedule": get_wsd_schedule,
}
