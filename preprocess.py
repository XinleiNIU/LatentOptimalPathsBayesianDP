import argparse

import yaml

from preprocessor.preprocessor import Preprocessor
import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    preprocessor = Preprocessor(config)
    if config['dataset'] == 'timit_pho':
        preprocessor.build_alignment()
    elif config['dataset'] == 'popcs':
         preprocessor.build_from_path_f0()
    else:
        preprocessor.build_from_path()
