#!/bin/bash
# Run the system test (see docs/INSTALL.md: System test for the reference output).
#
#   sbatch scripts/utils/run_system_test.sh   or   bash scripts/utils/run_system_test.sh
#
# Run from the repository root.

#SBATCH --job-name=test_system
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%A.out

mkdir -p logs

# activate your environment here, e.g.:  conda activate video-peft

python main.py --state_type test_system --config config/classification.yaml
