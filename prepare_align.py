import argparse

import yaml

from preprocessor import ryanspeech
from pdb import set_trace as st
import pdb

def main(config):
    if "ryanspeech" in config["dataset"]: 
        ryanspeech.prepare_align(config)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader) 

    main(config)
