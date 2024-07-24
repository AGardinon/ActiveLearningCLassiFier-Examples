# -------------------------------------------------- #
# Script for applying an active learning cycle
#
#
# AUTHOR: Andrea Gardin
# -------------------------------------------------- #

import yaml
import argparse
import scipy
import joblib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.metrics import log_loss, accuracy_score

import activeclf as alclf
from activeclf.learning import active_learning_cycle, get_starting_batch
from activeclf.utils.beauty import get_axes

# --- func

def recover_entropy(pdf: np.ndarray, decimals: int=2):
    try:
        entropy = np.around(scipy.stats.entropy(pdf, axis=1), decimals=decimals)
    except:
        entropy = np.around(scipy.stats.entropy(pdf.reshape(-1,1), axis=1), decimals=decimals)
    return entropy

def plot_line(x, y, axis):
    axis.plot(x, y, zorder=1)
    axis.scatter(x, y, c='0.', s=10, zorder=2)
    axis.grid(ls='--', alpha=.5)
    return axis



# --- main

def main(arguments):

    # read the config.yaml file provided
    exp_config = yaml.load(open(arguments.config, 'r'), Loader=yaml.FullLoader)

    # init output objects
    fileout_name = exp_config['experimentID']
    cross_entropy_list = list()
    accuracy_list = list()

    data = alclf.DataLoader(file_path=exp_config['dataset'],
                            target=exp_config['targetVariable'])
    
    # init the feature space for the active search
    data.feature_space(scaling=True)

    # - first batch, sampled random
    idxs = get_starting_batch(data=data.X, 
                              init_batch=exp_config['startingPoints'])

    # - init the functions to run the experiment
    if exp_config['kParam1'] and exp_config['kParam2']:
        print(f'Kernel initialized with values: A={exp_config['kParam1']}, B={exp_config['kParam2']}')
        kernel_function = exp_config['kParam1']*alclf.classification.RBF(exp_config['kParam2'])
    else:
        print(f'Kernel initialized with dafault values: A=1.0, B=1.0')
        kernel_function = 1.0*alclf.classification.RBF(1.0)

    # - start the ACLF experiment
    new_idxs = list()
    for cycle in tqdm(range(exp_config['Ncycles']+1), desc='Cycles'):

        idxs = idxs + new_idxs

        print(f'\n\n# ------------\n# --- Cycle {cycle}')
        print(f'ALpoints: {len(idxs)} / {len(data.X)}')

        print(f'Set up the Classifier and Acquisition function ..')
        classifier_func = alclf.ClassifierModel(model=exp_config['clfModel'],
                                                kernel=kernel_function,
                                                random_state=None)

        acquisition_func = alclf.DecisionFunction(mode=exp_config['acqMode'],
                                                  decimals=exp_config['entropyDecimals'],
                                                  seed=None)

        new_idxs, pdf = active_learning_cycle(
            feature_space=(data.X, data.y),
            idxs=idxs,
            new_batch=exp_config['newBatch'],
            clfModel=classifier_func,
            acquisitionFunc=acquisition_func,
            screeningSelection=exp_config['screeningSelection']
            )

        predicted_labels = classifier_func.clf.predict(X=data.X)
        accuracy_list.append(accuracy_score(y_true=data.y.to_numpy(dtype=int), y_pred=predicted_labels))

        if arguments.saveout:
            np.savetxt(fileout_name+f'_cycle{cycle}_idxs.idxs', idxs)
            np.savetxt(fileout_name+f'_cycle{cycle}_new_idxs.idxs', new_idxs)
            np.savetxt(fileout_name+f'_cycle{cycle}_pdf.pdf', pdf)
            joblib.dump(classifier_func.clf, fileout_name+f'_cycle{cycle}_classifier.pkl')

    if arguments.plots:
        fig, ax = get_axes(1,1)
        _ = plot_line(x=np.arange(exp_config['Ncycles']+1), y=accuracy_list, axis=ax)
        ax.set_title('Accuracy score')
        plt.show()

    print('\n# END')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', dest='config', required=True, type=str, help='Cycle configuration (.yaml file).')
    parser.add_argument('-saveout', dest='saveout', required=False, action='store_true', help='To save output files.')
    parser.add_argument('-plots', dest='plots', required=False, action='store_true', help='To plot output quantities with cycles.')
    args = parser.parse_args()
    main(arguments=args)