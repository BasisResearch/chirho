import subprocess

notebook_test_commands = [
    'pytest --nbmake -n=auto "./docs/source/"',
    'pytest --nbmake -n=auto "./docs/source/actual_causality/preemption_no_log_prob"',
    'pytest --nbmake -n=auto "./docs/source/actual_causality/pre_release_versions"',
]

for command in notebook_test_commands:
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Notebook test command failed: {e}")
