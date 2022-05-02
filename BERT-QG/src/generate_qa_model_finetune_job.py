import argparse, sys

def main(args):
    
    with open(f'job_script_{args.exp_name}.job', 'w') as out_file:
        out_file.write('#!/bin/bash\n')
        out_file.write(f'#SBATCH --job-name={args.exp_name}\n')
        out_file.write('#SBATCH --account=project_2001403\n')
        out_file.write('#SBATCH --partition=gpu\n')
        out_file.write(f'#SBATCH --time={args.max_hours}:00:00\n')
        out_file.write('#SBATCH --ntasks=1\n')
        out_file.write('#SBATCH --cpus-per-task=10\n')
        out_file.write('#SBATCH --mem-per-cpu=8000\n')
        out_file.write(f'#SBATCH --gres=gpu:v100:{args.gpus}\n')
        out_file.write('#SBATCH --mail-user=ilmari.kylliainen@helsinki.fi\n')
        out_file.write(f'#SBATCH --output={args.exp_name}.log\n')
        out_file.write('#SBATCH --mail-type=BEGIN,END,FAIL\n\n')
        out_file.write('conda activate qgen\n\n')
        out_file.write(f'srun python train-causal-hlsqg.py --cfg configs/train/{args.config_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate SLURM job scripts')
    parser.add_argument("--exp_name", required=True, help="Name for experiment and output model")
    parser.add_argument("--max_hours", required=True, type=int, choices=range(1,50), metavar="[1-50]",
                        help="Max length for the job in hours")
    parser.add_argument("--gpus", required=True, type=int, choices=range(1,9), metavar="[1-9]",
                        help="Num of GPUs for job")
    parser.add_argument("--config_file", required=True,
                        help="Config file")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    main(args)