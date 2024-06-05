import os
import sys
import yaml
cwd = '/teamspace/studios/this_studio/letsdoit'
if (cwd not in sys.path):
    sys.path.append(cwd)
sys.path.append(os.path.dirname(cwd))
from pipeline.pipeline import Pipeline


path_config = '/teamspace/studios/this_studio/letsdoit/config/config.yml'

def main():

    with open(path_config, 'r') as file:
        cfg = yaml.safe_load(file)

    pipe = Pipeline(**cfg)

    pipe.run()
    print('finish!')

if __name__ == '__main__':
    main()