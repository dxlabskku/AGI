import os
import glob

root_path = '/data/jupyter/AGI/datasets/XD-violence/test/video'    ## the path of features
files = sorted(glob.glob(os.path.join(root_path, "*.mp4")))
violents = []
normal = []
with open('/data/jupyter/AGI/encoders/multi_modal/video_audio/test.list', 'w+') as f:  ## the name of feature list
    for file in files:
        if '_label_A' in file:
            normal.append(file)
        else:
            newline = file+'\n'
            f.write(newline)
    for file in normal:
        newline = file+'\n'
        f.write(newline)