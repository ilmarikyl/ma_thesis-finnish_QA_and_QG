import argparse, sys



def main(args):
    
    with open(f'job_script_{args.exp}.job', 'w') as out_file:
        out_file.write('#!/bin/bash\n')
        out_file.write(f'#SBATCH --job-name={args.exp}\n')
        out_file.write('#SBATCH --account=project_2001403\n')
        out_file.write('#SBATCH --partition=gpu\n')
        out_file.write(f'#SBATCH --time=10:00:00\n')
        out_file.write('#SBATCH --ntasks=1\n')
        out_file.write('#SBATCH --cpus-per-task=10\n')
        out_file.write('#SBATCH --mem-per-cpu=8000\n')
        out_file.write(f'#SBATCH --gres=gpu:v100:1\n')
        out_file.write('#SBATCH --mail-user=ilmari.kylliainen@helsinki.fi\n')
        out_file.write(f'#SBATCH --output={args.exp}.log\n')
        out_file.write('#SBATCH --mail-type=BEGIN,END,FAIL\n\n')
        out_file.write('module purge\n\n')
        out_file.write('module load pytorch/1.9\n\n')
        out_file.write(f'srun python output_gpt-qa-model_predictions.py /scratch/project_2001403/ilmariky/FinGPT-QA-v1/{args.model}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate SLURM job scripts for GPT-QA model evaluation')

    parser.add_argument('model', metavar='model_dir', help='Model to use for creating the predictions.')
    parser.add_argument("--exp", required=True, help="Name for experiment and output model")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()



    main(args)