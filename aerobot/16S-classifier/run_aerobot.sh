#!/bin/bash
#SBATCH --job-name=run_classifier    # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=goldford@caltech.edu     # Where to send mail	
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --nodes=1
#SBATCH --mem=16gb                     # Job memory request
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00              # Time limit hrs:min:sec
#SBATCH --output=run_classifier%j.log   # Standard output and error log

python aerobot-16s.py --fasta_file seqs.train.fasta --weights_file models/clf_model_best.pt --output_file predictions.TrainingData.csv --probabilities_file predictionProbs.TrainingData.csv
python aerobot-16s.py --fasta_file seqs.test.fasta --weights_file models/clf_model_best.pt --output_file predictions.TestingData.csv --probabilities_file predictionProbs.TestingData.csv