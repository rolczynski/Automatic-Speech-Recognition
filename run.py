import argparse
import os
from source.deepspeech import DeepSpeech
from source.configuration import Configuration

abspath = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(abspath)
os.chdir(ROOT_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--configuration', help='Experiment configuration')
    args = parser.parse_args()

    # Read configuration file
    config = Configuration(file_path=args.configuration)
    # Set up DeepSpeech object
    ds = DeepSpeech(config)

    # Model optimization
    ds.train()
    # Save whole deepspeech model
    ds.save()
