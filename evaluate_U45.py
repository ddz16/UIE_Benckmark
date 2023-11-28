import os
import numpy as np
import argparse
from PIL import Image
from myutils.quality_no_refer import calculate_path_NRIQA

parser = argparse.ArgumentParser(description='Evaluating U45 dataset')

parser.add_argument('--method_name', type=str, default='UIEC2Net', 
                    help='method name, any subfolder in ./data/U45/All_Results/')

hparams = parser.parse_args()

test_path = "./data/U45/All_Results/" + hparams.method_name  # the pred images

calculate_path_NRIQA(test_path)
