# Time-series Prediction from Limit Order Book Data


<div style="display: flex;">
    <div style="flex: 50%; padding-left: 10px;">
        This repository contains code and instructions for time-series prediction on limit order book data using deep learning models. The goal is to provide a comprehensive, yet easy-to-use, framework for data preparation, model training, and evaluation.
    </div>
    <div style="flex: 30%;">
        <img src="assets/dalle_image.png" alt="DALL-E generated image" style="width: 100%;">
        <p style="text-align: center;">Credits to ChatGPT</p>
    </div>
</div>


## 1. Setup

First, clone this repo

```bash
git clone https://github.com/pietrolesci/time-series-prediction.git
```

To manage the environment we use [uv](https://docs.astral.sh/uv/getting-started/installation/). If you do not have it installed, simply run

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

then, exactly reproduce the Python environment by running

```bash
uv sync
```



## 2. Data

Put the data in the data folder. We expect it to be in a parquet format. Then, run

```bash
uv run scripts/prepare_data.py
```

This script will prepare the data for training, split it into train, validation, and test splits, and store them as memory mapped numpy files (named `{train, val, test}_memmap.npy`) and the associated metadata (named `{train, val, test}_memmap.meta`). This allows us to efficiently load it during training. The data splitting process is done as follows:

- The initial 80% (2.8M instances) are used for training

- A gap of 100k instances is left between training and validation to ensure robust model evaluation. Specifically, if the training data ends too close to the start of the validation data, there may be subtle overlap or dependencies (e.g., autocorrelations, seasonality, and any short-term patterns) that the model can exploit. This can lead to overly optimistic validation results.

- The subsequent 7% (250k) instances are used for validation

- Again, a gap of 50k instances is left between validation and test split, since the validation set is used for model selection (e.g., early stopping)

- Finally, 8.5% (300k) instances are used for testing

In total, the gap (unused instances) amounts to 4.5% of the data.




## 3. Training

To start training run

```bash
uv run scripts/train.py model=deeplob  # or picodeeplob
```

The experiment configuration is handled with [hydra](https://hydra.cc/). So, you can simply pass arguments in the command line (e.g., `model=deeplob`). To run multiple experiments in parallel with joblib you can run

```bash
uv run scripts/train.py model=deeplob optim.lr=0.01,0.001 +launcher=joblib
```

Currently, we support two models: DeepLOB ([DeepLOB: Deep Convolutional Neural Networks for Limit Order Books](https://arxiv.org/abs/1808.03668)) and PicoDeepLOB, which is a modification of DeepLOB that replaces the LSTM component with a decoder-only transformer based on the Llama2 architecture. The code allows to configure the task as a regression or a classification task. Initial experiments show that the classification format is more challenging and leads to underperforming models.

Optimisation is performed using the AdamW optimiser and multiple learning rate schedules are supported. New optimisers can be seemlessly added.

During training, we log many useful statistics: MSEloss, R2 score, distribution of the model predictions (mean, standard deviation, minimum, and maximum), gradient norms, learning rate, and time (e.g., batch processing time).

The artefacts from a training run (e.g., metadata, checkpoints, logs, tensorboard logs, etc) are stored in a folder named `{model name}_{datetime}` (e.g., `deeplob_2024-12-10T18-39-58`) under the `outputs/model_train` folder (you can change from `model_train` to anything by passing `out_parent_folder=<your_path>`).

Importantly, we use Tensorboard as our logger. Additionallly, at the end of each experiment, the tensorboard logs are also exported as a parquet file (named `tb_logs.parquet`) which allows easier analysis.

Finally, we write to disk the model predictions every time we run validation (additionally, training predictions can be stored too, but it's usually too big -- that's why we only log their distribution during training).




## Evaluation

Evaluation scores are directly computed by the training scripts. Specifically, during training, we run the evaluation 4 times per epoch on both the validation and the test splits. Note: the test split is not used for model selection! We only add this to check how much our validation and test scores correlate. This is useful to know because once we pin down the correct training recipe, to create the model artefact to submit, we will use 90% of the data as training and only the last 10% for model selection; in this way we use as much data as possible to train prior to submitting.

Additionally, we can evaluate any checkpoint by running (note that in this case we use simply `argparse` for the CLI interface)

```bash
uv run scripts/eval.py --checkpoint <checkpoint_path> --out_dir data/predictions
```

As a result, under the specified `out_dir`, we get a file named `{checkpoint}.parquet` with the model predictions and the actual target value which allows for easier analysis. 



