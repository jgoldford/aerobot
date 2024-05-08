#!/bin/bash
#SBATCH --job-name=ft_16s_classifier    # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=goldford@caltech.edu     # Where to send mail	
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --nodes=1
#SBATCH --mem=16gb                     # Job memory request
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00              # Time limit hrs:min:sec
#SBATCH --output=ft_16s_classifier%j.log   # Standard output and error log

python fine-tune-16s-classification.py