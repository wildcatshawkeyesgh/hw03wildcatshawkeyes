# HW03 — wildcatshawkeyes

## Setup

Clone the repo and enter the directory

Create virtual environment and install dependencies: uv sync

## Running the scripts

Note: Ensure training script data is on the same path on lovelace for me as it is for you.

### ImageNet CNN training

Command to run imagenet script.
uv run python scripts/imagenet_impl.py

To run training in the background:
cd scripts
./run_bg.sh "uv run python imagenet_impl.py --epochs=10000 --train_ratio=0.0001 --val_ratio=0.0001"
(you must ensure the run_bg.sh is executable I am unsure if an executable script uploaded to github stays executable)

### ImageNet inference onnx model
In order to run imagenet_inference you must change to the scripts directory

Then run:uv run imagenet_inference.py

### ACC classifier training

To run:
uv run python scripts/acc_classifier.py
To run in the background:

cd scripts
./run_bg.sh "uv run python acc_classifier.py"

### ACC classifier inference onnx model

The trained ACC model is saved as `acc_model.onnx` in the main directory.
