DIR="/scratch/ssd001/home/kaselby/fsdm/runs"

import os
def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def set_folder():
    mkdirs(DIR)
    dct={"dir": DIR}
    return dct["dir"]
