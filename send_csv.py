from sys import argv
from experiment.experiment import send_collected_csv


if __name__ == '__main__':
    result_dir = argv[0]
    send_collected_csv(result_dir=result_dir)
