import argparse
import numpy as np
import util

def parse_cmd_line() -> argparse.Namespace:
    """
    Parses command-line arguments.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--force_runs', nargs='+', type=int, required=True)
    parser.add_argument('--no_force_runs', nargs='*', type=int, default=[])
    parser.add_argument('--use_acceleration', action='store_true', default=False)
    parser.add_argument("--overfit", action='store_true', default=False)
    return parser.parse_args()


def analyze_nonzero_chunks(signal_list, threshold = 0, min_chunk_length = 10):
    """
    Analyzes non-zero chunks in a list of (signal_length, 3) arrays.
    
    Parameters:
        signal_list (list of np.ndarray): List of arrays with shape (signal_length, 3).
        
    Returns:
        dict: A dictionary containing mean, std, min, and max of non-zero chunk lengths.
    """
    chunk_lengths = []
    for signal in signal_list:
        for dim in range(signal.shape[1]):  # Iterate over the 3 dimensions
            signal[signal < threshold] = 0  # Thresholding
            non_zero_mask = signal[:, dim] != 0  # Boolean mask of non-zero values
            diff = np.diff(np.concatenate(([0], non_zero_mask.astype(int), [0])))
            start_indices = np.where(diff == 1)[0]  # Start of non-zero chunks
            end_indices = np.where(diff == -1)[0]  # End of non-zero chunks
            lengths = end_indices - start_indices
            chunk_lengths.extend(lengths[lengths > min_chunk_length])  # Only consider chunks > 5

    if not chunk_lengths:
        return {"mean": 0, "std": 0, "min": 0, "max": 0}  # Handle empty case

    chunk_lengths = np.array(chunk_lengths)

    return {
        "mean": np.mean(chunk_lengths),
        "std": np.std(chunk_lengths),
        "min": np.min(chunk_lengths),
        "max": np.max(chunk_lengths),
        "median": np.median(chunk_lengths),
        "percentile_25": np.percentile(chunk_lengths, 25),
        "percentile_75": np.percentile(chunk_lengths, 75)
    }

def main():
    """
    Main function to load datasets and process them.
    """
    args = parse_cmd_line()
    run_nums = util.get_run_numbers(args)
    data_dir = "data"
    sets = ["train", "test"]

    for s in sets:
        features, targets, _, _ = util.load_dataset(
            path=data_dir,
            force_policy_runs=run_nums[s][0],
            no_force_policy_runs=run_nums[s][1],
            sequential=True,
            use_acceleration=args.use_acceleration,
            create_plots=True,
            crop_runs=True
        )
        print(analyze_nonzero_chunks(targets))

if __name__ == "__main__":
    main()
