import logging
import time
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch
from lightning.fabric import Fabric, seed_everything
from tqdm.auto import tqdm

from src.datamodule import DataModule
from src.models import MODEL_CONFIG

# Configure the logger and configure colorlog
logger = logging.getLogger("eval")


# In the LightningModule, the model is saved at the attribute self.model
# which is then used as a key in the state_dict stored in the checkpoint.
# When loading the model as a standalone torch.nn.Module, we need to remove
# the prefix "model." from the keys in the state_dict.
LIGHTNING_MODEL_ATTR = "model."


def load_model_and_data_from_pl(checkpoint_path: str | Path) -> tuple[MODEL_CONFIG, torch.nn.Module, DataModule]:
    logger.info(f"Reading checkpoint from {checkpoint_path=}")
    checkpoint = torch.load(str(checkpoint_path), weights_only=False)

    # Create config object
    config = checkpoint["hyper_parameters"]["config"]
    logger.info(f"Model {config=}")

    # Load re-initialised model
    model = config.get_model()
    state_dict = {k.removeprefix(LIGHTNING_MODEL_ATTR): v for k, v in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)

    # Load datamodule
    datamodule = DataModule(**checkpoint["datamodule_hyper_parameters"])

    return config, model, datamodule


@torch.inference_mode()
def predict(model: torch.nn.Module, batch: dict) -> list[tuple[float, float]]:
    x, y = batch["X"], batch["y"]
    preds = model(x)

    out = {"preds": preds, "y": y, "idx": batch["idx"]}

    for k, v in out.items():
        if v.dim() > 1:
            v = v[..., -1]
        out[k] = v.cpu().numpy().tolist()

    return list(zip(*out.values(), strict=True))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch_compile", action="store_true", default=False)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32-true")
    parser.add_argument("--out_dir", type=str, default="data/predictions")
    args = parser.parse_args()

    start_time = time.time()
    logger.info(f"Seed enabled: {args.seed}")
    ck_path = Path(args.checkpoint)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model and config
    config, model, datamodule = load_model_and_data_from_pl(ck_path)

    # Prepare model
    fabric = Fabric(accelerator=args.accelerator, precision=args.precision)
    model = fabric.setup_module(model)
    if args.torch_compile:
        model.compile()
    model.eval()
    datamodule.setup("test")

    # Run inference
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    for dl_idx, dl in enumerate(datamodule.val_dataloader()):
        dl_name = "val" if dl_idx == 0 else "test"

        # can keep it in memory
        predictions = []
        for batch in tqdm(dl, desc=f"Running dataloader={dl_name}"):
            batch = fabric.to_device(batch)
            out = predict(model, batch)
            predictions += out

        # Save predictions
        df = pd.DataFrame(predictions, columns=["pred", "y", "idx"])
        df.to_parquet(out_dir / f"{ck_path.parents[1].name}_{dl_name}.parquet", index=False)

    logger.info(f"Total time: {(time.time() - start_time) // 60} minutes")
