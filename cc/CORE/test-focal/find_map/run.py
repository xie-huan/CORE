import yaml
import subprocess

bugs_info = None
with open("/root/ase/test-focal/find_map/bugs.yaml", 'r') as stream:
    try:
        bugs_info = yaml.safe_load(stream)
        a = 1
    except Exception as exc:
        print(exc)

project_dir="/mnt/project/"
output_dir="/mnt/tests2focal/"
for program, versions in bugs_info.items():
    for version in versions:
        bug_id=program+"_"+str(version)+"_buggy"

        # params = [program, str(version)]
        # --repo_id Chart_1_buggy --grammar ~/find_map/java-grammar.so --tmp ~/tmp/tmp/ --output ~/tmp/output
        result = subprocess.run(['python3','/root/ase/test-focal/find_map/find_map.py', 
                                 '--repo_id', bug_id, 
                                 '--grammar', '/root/ase/test-focal/find_map/java-grammar.so',
                                 '--tmp', project_dir,
                                 '--output', output_dir
                                 ], capture_output=True, text=True)
        print(result.stdout)