import subprocess

program_list = ['best_challenge.py', 'best_solution.py']

for program in program_list:
    subprocess.call(['python', 'program'])
    print(f'Finished: {program}')