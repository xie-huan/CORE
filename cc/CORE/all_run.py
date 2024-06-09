import yaml
import subprocess

bugs_info = None
with open("bugs.yaml", 'r') as stream:
    try:
        bugs_info = yaml.safe_load(stream)
        a = 1
    except Exception as exc:
        print(exc)

for program, versions in bugs_info.items():
    for version in versions:
        params = [program, str(version)]
        result = subprocess.run(['./run.sh']+params, capture_output=True, text=True)
        print(result.stdout)