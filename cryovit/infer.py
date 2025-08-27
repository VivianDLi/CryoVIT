import argparse
from pathlib import Path

from cryovit.run.infer_model import run_trainer

def run_inference():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference given a data folder (or text file specifying files).")
    parser.add_argument("data", type=str, required=True, help="Directory or .txt file of tomograms")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--result_dir", type=str, default=None, required=False, help="Path to the directory to save prediction results. If not provided, adds results to the input data.")

    args = parser.parse_args()
    data = Path(args.data)
    model_path = Path(args.model_path)
    result_dir = Path(args.result_dir) if args.result_dir else None

    ## Sanity Checking
    assert data.exists(), "Test data path does not exist."
    assert model_path.exists(), "Model path does not exist."
    assert model_path.suffix == ".model", "Model path must point to a .model file."
    assert result_dir is None or (result_dir.exists() and result_dir.is_dir()), "Specified result_dir must be a directory."

    ## Converting everything into formats recognizable by the inference function
    
    ## Creating Hydra configurations

    ## Running inference