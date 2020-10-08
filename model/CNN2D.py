import argparse
import evaluate_models as eval

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='CNN demo.'
    )
    parser.add_argument(
        '-a', '--algorithm',
        default = 'CNN2D'
        help='Model architecture identifier'
    )
    parser.add_argument('data_dir', help='Base directory for run parameters.')
    parser.add_argument('output_dir', help='Base directory for run results.')
    


    eval.main(args=parser.parse_args())
