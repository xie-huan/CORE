"""
    -p "motivation" -i "artificial_bug" -m "GP02" -e "origin"

    # supported experiment, i.e., -e
    # origin, resampling, undersampling, fs, cvae, fs_cvae
    # "origin": no other argument, e.g., -d "manybugs" -p motivation -i "artificial_bug" -m "GP02" -e "origin"
    # "resampling": no other argument
    # "undersampling": no other argument
    # "fs": component_percent, eigenvalue_percent, e.g., -d "manybugs" -p motivation -i "artificial_bug" -m "GP02" -e "fs" -cp 0.7 -ep 0.7
    # "cvae": no other argument
    # "fs_cvae": like "fs"
"""


def find(args, arg):
    try:
        i = args.index(arg)
        return args[i + 1]
    except:
        raise Exception(f"Missing required argument: {arg}")


def parse_args(args):
    args = args[1:]

    config_dict = {}
    required_args = ["-d", "-p", "-i", "-m", "-e"]
    for arg in required_args:
        config_dict[arg] = find(args, arg)

    if ((config_dict["-e"] == "origin") or (config_dict["-e"] == "resampling") or (
            config_dict["-e"] == "undersampling") or (config_dict["-e"] == "cvae")) and len(config_dict) != 5:
        raise Exception(f"{config_dict['-e']} has no -cp or -ep")
    if config_dict["-e"] == "fs" or config_dict["-e"] == "fs_cvae":
        config_dict["-cp"] = find(args, "-cp")
        config_dict["-ep"] = find(args, "-ep")
    if config_dict["-e"] not in ["origin", "resampling", "undersampling", "fs", "cvae", "fs_cvae", "smote"]:
        raise Exception(f"Wrong parameters {config_dict}, please check again.")

    optional_args = ["-r", "-a"]
    for arg in optional_args:
        if arg in args:
            config_dict[arg] = args[args.index(arg) + 1]

    return config_dict
