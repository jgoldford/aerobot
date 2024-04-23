'''A wrapper for running scripts (phylo_bias.py or run.py) on each feature type and model class. This should be run on the HPC,
which has slurm installed.'''

import subprocess
from aerobot.io import FEATURE_TYPES
import os

# TODO: Might be worth supporting command-line arguments for this.


if __name__ == '__main__':
    # Get the path within aerobot where all the Python scripts are stored. 
    SCRIPTS_PATH, _ = os.path.split(os.path.abspath(__file__))
    SCRIPT = os.path.join(SCRIPTS_PATH, 'phylo-bias.py')
    OUTPUT_DIR = '~'
    TIME = '10:00:00' # Time for slurm job. 
    MEM = '20GB' # Memory for slurm job. 

    for model_class in ['logistic', 'nonlinear']:
        for feature_type in FEATURE_TYPES:
            output_filename = f'run_results_{model_class}_{feature_type}.json'
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            cmd = f'python {SCRIPT} {model_class} --feature-type {feature_type} -o {output_path}'
            # Submit the slurm job. 
            subprocess.run(f'sbatch --wrap "{cmd}" -N 1 --time {TIME} --mem {MEM}', check=True, shell=True)