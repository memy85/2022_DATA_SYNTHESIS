
#%%

import pandas as pd
from pathlib import Path
import os
import argparse
import matplotlib.pyplot as plt
import scienceplots

projectPath = Path().cwd()
# projectPath = Path(__file__).absolute().parents[2]
print(projectPath)

os.sys.path.append(projectPath.as_posix())

from src.MyModule.utils import *

#%%

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', default = 0, type = int)
    args = parser.parse_args()

    return args
#%%

def load_privacy_test_result(randomseed):
    '''
    you should change the code if there are any randomseed required  
    '''

    datapath = projectPath.joinpath("figures/privacy_test1.csv")
    data = pd.read_csv(datapath)

    return data

#%%

def parse_data(data) :
    baseline = data.loc[1][1].split('/')[0].strip()[1:-1]
    config = load_config()
    epsilon = config['epsilon']

    numbers = []
    numbers.append(float(baseline))
    for eps in epsilon :
        loc = data.loc[1].get(f'epsilon{eps}').find('[', 1)
        number = data.loc[1].get(f'epsilon{eps}')[loc+1:-1]
        number = float(number)

        numbers.append(number)
    print("this is numbers : {}".format(numbers))
    return numbers


#%%

def plot(data, figpath):

    config  = load_config()
    epsilons = config['epsilon']
    plt.style.use(['science', 'ieee'])

    fig, ax = plt.subplots(1,1)

    privacyScore = data[1:]
    xs = [f"{eps}" for eps in epsilons]
    
    ax.plot(xs, privacyScore, '.-')
    ax.axhline(data[0], color = 'red')
    ax.set_xlabel('epsilons ($\epsilon$)')
    ax.set_ylabel('Reidentification Score')
    ax.legend(['Synthetic Data', 'Baseline'])
    plt.show()
    plt.savefig(figpath.joinpath('privacy_test.png'), dpi = 500)

    return 0

#%%


def main():

    args = parse_arguments()
    figpath = projectPath.joinpath('figures')

    data = load_privacy_test_result(args.random_seed)
    parsed_privacy_score = parse_data(data)
    
    plot(parsed_privacy_score, figpath)
    print('finished!')


if __name__ == "__main__" :
    main()
