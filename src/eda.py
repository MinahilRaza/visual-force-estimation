import argparse
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
            crop_runs=False
        )

if __name__ == "__main__":
    main()
