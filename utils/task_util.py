import os


def task_complete(message: str):
    command = f"""~/imail.py "{message}"
    """
    os.system(command)