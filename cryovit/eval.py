import argparse
from pathlib import Path

from cryovit.run.eval_model import run_trainer

def run_evaluation():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation given a test folder (or text file specifying files).")
    parser.add_argument("test_data", type=str, required=True, help="Directory or .txt file of test tomograms")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--result_dir", type=str, default=None, required=False, help="Path to the directory to save the evaluation results: a .csv file containing evaluation metrics. Defaults to the current directory.")
    parser.add_argument("--visualize", "-v", action="store_true", help="If set, will save visualizations of the predictions in the result_dir folder.")

    args = parser.parse_args()
    test_data = Path(args.test_data)
    model_path = Path(args.model_path)
    result_dir = Path(args.result_dir) if args.result_dir else Path.cwd()

    ## Sanity Checking
    assert test_data.exists(), "Test data path does not exist."
    assert model_path.exists(), "Model path does not exist."
    assert model_path.suffix == ".model", "Model path must point to a .model file."
    assert result_dir.exists() and result_dir.is_dir(), "Result directory either does not exist or isn't a directory."

    ## Converting everything into formats recognizable by the evaluation function
    
    ## Creating Hydra configurations

    ## Running evaluation