'''A wrapper for the run_model.py script for training and evaluating multiple models on the HPC.'''

import subprocess
from aerobot.io import FEATURE_TYPES
import os

# TODO: Might be worth supporting command-line arguments for this.

# Get the path within aerobot where all the Python scripts are stored. 
SCRIPTS_PATH, _ = os.path.split(os.path.abspath(__file__))
RUN_MODEL = os.path.join(SCRIPTS_PATH, RUN_MODEL)
OUTPUT_DIR = '~'
OUTPUT_FORMAT = 'json'
TIME = '02:00:00' # Time for slurm job. 
MEM = '20GB' # Memory for slurm job. 


if __name__ == '__main__':

    for model_class in ['logistic', 'nonlinear']:
        for feature_type in FEATURE_TYPES:
            output_filename = f'run_model_results_{model_class}_{feature_type}.' + OUTPUT_FORMAT
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            cmd = f'{RUN_MODEL} {model_class} --feature-type {feature_type} --output-format {OUTPUT_FORMAT} -o {output_path}'
            # Submit the slurm job. 
            subprocess.run(f'sbatch --wrap "{cmd}" -N 1 --time {TIME} --mem {MEM}', check=True, shell=True)