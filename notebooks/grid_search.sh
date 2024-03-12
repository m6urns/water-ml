#!/bin/bash
#SBATCH --job-name=grid_search     # Job name
#SBATCH --output=grid_search.out   # Output file
#SBATCH --error=grid_search.err    # Error file
#SBATCH --time=24:00:00            # Time limit hrs:min:sec
#SBATCH --partition=batch          # Partition to submit to
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks (processes)
#SBATCH --cpus-per-task=16         # Number of CPU cores per task
#SBATCH --mem=32G                  # Total memory limit

# Load any necessary modules or activate environments
# Example: module load python/3.8
# or
# Example: source activate myenv
source /opt/miniconda3/bin/activate
conda activate water-ml

# Navigate to your project directory (if necessary)
cd /home/mlister/water-ml/notebooks

# Execute the script
python param_grid_ExtraTreesClassifier.py

conda deactivate