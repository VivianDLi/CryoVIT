import argparse
from pathlib import Path

from cryovit.utils import id_generator
from cryovit.run.train_model import run_trainer

def run_training():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model training given a training and optional validation folder (or text file specifying files).")
    parser.add_argument("train_data", type=str, required=True, help="Directory or .txt file of training tomograms")
    parser.add_argument("--val_data", type=str, required=False, help="Directory or .txt file of validation tomograms")
    parser.add_argument("--model", type=str, required=False, default="cryovit", choices=["cryovit", "unet3d", "sam2", "medsam"], help="Type of model to train")
    parser.add_argument("--result_path", type=str, default=None, required=False, help="Path to save the training results: a .zip file containing model weights and metadata. Defaults to the current directory.")
    parser.add_argument("--name", type=str, required=False, default=None, help="Name to identify the model. If not provided, a random name will be generated.")

    args = parser.parse_args()
    train_data = Path(args.train_data)
    val_data = Path(args.val_data) if args.val_data else None
    model_type = args.model
    if args.name:
        model_name = args.name
    else:
        model_name = model_type + id_generator()
    result_path = Path(args.result_path) if args.result_path else Path.cwd() / f"{model_name}.model"
    
    ## Sanity Checking
    assert train_data.exists(), "Training data path does not exist."
    if val_data is not None:
        assert val_data.exists(), "Validation data path does not exist."
        
    ## Converting everything into formats recognizable by the training function
    
    ## Creating Hydra configurations
    
    ## Running training