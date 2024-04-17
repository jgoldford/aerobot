'''Code for generating figures from model outputs. Functions are designed to interface with results dictionaries, which are given as 
output by model training and evaluation scripts.'''
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import matplotlib.ticker as ticker
from typing import Dict, NoReturn, List
import os

# TODO: Work this in to the io.py module. Also add other feature types. 
PRETTY_NAMES = {'KO':'All gene families', 'embedding.geneset.oxygen':'Five-gene set', 'chemical':'Chemical features', 'aa_1mer':'Amino acid counts', 'aa_3mer':'Amino acid trimers'}

# Some specs to make plots look nice. 
TITLE_FONT_SIZE, LABEL_FONT_SIZE = 12, 10
FIGSIZE = (4, 3)
PALETTE = 'Set1'
# Set all matplotlib global parameters.
plt.rc('font', **{'family':'sans-serif', 'sans-serif':['Arial'], 'size':LABEL_FONT_SIZE})
plt.rc('xtick', **{'labelsize':LABEL_FONT_SIZE})
plt.rc('ytick', **{'labelsize':LABEL_FONT_SIZE})
plt.rc('axes',  **{'titlesize':TITLE_FONT_SIZE, 'labelsize':LABEL_FONT_SIZE})

COLORS = ['tab:gray', 'tab:green', 'tab:blue']

# TODO: Update this for all feature types. 
ANNOTATED = ['KO', 'embedding.geneset.oxygen'] 
UNANNOTATED = ['chemical', 'aa_1mer', 'aa_3mer']

def plot_training_curve(results:Dict, path:str=None) -> NoReturn:
    '''Plot the Nonlinear classifier training curve. Save the in the current working directory.'''
    assert results['model_class'] == 'nonlinear', 'plot_training_curve: Model class must be Nonlinear.'
    assert 'training_losses' in results, 'plot_training_curve: Results dictionary must contain training losses.'

    # Extract some information from the results dictionary. 
    train_losses = results.get('training_losses', [])
    train_accs = results.get('training_acss', [])
    val_losses = results.get('validation_losses', [])
    val_accs = results.get('validation_accs', [])
    feature_type = results['feature_type']

    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx() # Create another axis for displaying accuracy.
    loss_ax.set_title(f'{PRETTU_NAMES[feature_type]} training curve') # Set the title.

    lines = loss_ax.plot(train_losses, c=COLORS[0], label='training loss')
    lines += loss_ax.plot(val_losses, c=COLORS[0], linestyle='--', label='validation loss')
    lines += acc_ax.plot(val_accs, linestyle='--', c=COLORS[1], label='validation accuracy')
    lines += acc_ax.plot(train_accs, c=COLORS[1], label='training accuracy')

    loss_ax.set_ylabel('MSE loss')
    loss_ax.set_xlabel('epoch') # Will be the same for both axes.
    acc_ax.set_ylabel('balanced accuracy')
    acc_ax.set_ylim(top=1, bottom=0)

    acc_ax.legend(lines, ['training loss', 'validation loss', 'validation accuracy', 'training accuracy'])

    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=500, format='PNG', bbox_inches='tight')
        plt.close()  # Prevent figure from being displayed in notebook.
    else:
        plt.show()


def _format_barplot_axes(ax:plt.Axes, feature_types:List[str]=None, binary:bool=False):

    random_baseline = 0.5 if binary else 0.33 # Expected performance for random classifier on task. 

    # Label bins with the feature name. 
    ax.set_xticks(np.arange(0, len(feature_types), 1), [PRETTY_NAMES[f] for f in feature_types], rotation=45, ha='right')
    
    # Set up left y-axis with the balanced accuracy information. 
    ax.set_ylabel('balanced accuracy')
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1)) # xmax is the number to divide by for the percentage.
    # Add a second set of y-ticks on the right to indicate percentage performance increase over random (33% accurate)
    new_ax = ax.twinx()
    new_ax.set_ylim(0, 1)
    yticks = np.round(100 * (np.arange(0, 1.1, 0.1) - random_baseline) / random_baseline, 0)
    ytick_labels = [f'{v:.0f}%' for v in yticks]
    new_ax.set_yticks(yticks, ytick_labels)
    new_ax.set_ylabel('percent above random')

    # Add horizontal line marking model performance with random classification. 
    ax.axhline(random_baseline, color='grey', linestyle='--', linewidth=2, zorder=-10)

    plt.sca(ax) # Just in case creating the new axis messes things up. 


def plot_model_accuracy_barplot(results:Dict[str, Dict]=None, path:str=None) -> NoReturn:

    # Two bars per model, one for training accuracy and one for validation accuracy
    fig, ax = plt.subplots(1, figsize=(7, 3))

    feature_types = list(results.keys())
    # Extract the final balanced accuracies on training and validation sets from the results dictionaries. 
    train_accs  = [results[feature_type]['training_acc'] for feature_type in feature_types]
    val_accs  = [results[feature_type]['validation_acc'] for feature_type in feature_types]
    binary = results[feature_types[0]]['binary'] # Assume all have the same value, but might want to add a check.

    plt.title('Ternary classification' if not binary else 'Binary classification', loc='left')
    
    colors = [COLORS[1] if f in UNANNOTATED else COLORS[2] for f in feature_types] # Map annotation-free or -full features to different colors. 
    ax.bar(np.arange(0, len(feature_types), 1) - 0.2, train_accs, width=0.4, label='training', color=colors, edgecolor='k', linewidth=0.5, hatch='//')
    ax.bar(np.arange(0, len(feature_types), 1) + 0.2, val_accs, width=0.4, label='validation', color=colors, edgecolor='k', linewidth=0.5)

    # Custom legend. Colors indicate annotation-free or annotation-full, and hatching indicates training or validation set. 
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='k', linewidth=0.5, hatch='////')]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='k', linewidth=0.5, hatch=''))
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[1], edgecolor='k', linewidth=0.5))
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[2], edgecolor='k', linewidth=0.5))
    labels = ['training', 'validation', 'annotation-free', 'with annotation']
    plt.legend(handles, labels, ncol=2, fontsize=7, columnspacing=0.3, handletextpad=0.3, loc='upper left', bbox_to_anchor=(0.25, 0.99))

    _format_barplot_axes(ax, feature_types=feature_types, binary=binary)

    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=500, format='PNG', bbox_inches='tight')
        plt.close()  # Prevent figure from being displayed in notebook.
    else:
        plt.show()


def plot_model_comparison_barplot(nonlinear_results:Dict[str, Dict], logistic_results:Dict[str, Dict], path:str=None) -> NoReturn:

    fig, ax = plt.subplots(1, figsize=(7, 3))

    feature_types = list(nonlinear_results.keys())
    for feature_type in feature_types:
        assert feature_type in logistic_results, f'plot_nonlinear_logistic_comparison_barplot: {feature_type} is missing in the logistic regression results.'
    
    # Extract the final balanced accuracies on training and validation sets from the results dictionaries. 
    nonlinear_val_accs  = [nonlinear_results[feature_type]['validation_acc'] for feature_type in feature_types]
    logistic_val_accs  = [logistic_results[feature_type]['validation_acc'] for feature_type in feature_types]
    binary = nonlinear_results[feature_types[0]]['binary'] # Assume all have the same value, but might want to add a check.
    
    plt.title('Ternary classification' if not binary else 'Binary classification', loc='left')

    ax.bar(np.arange(0, len(feature_types), 1) - 0.2, logistic_val_accs, width=0.4, label='logistic', color=COLORS[0], edgecolor='k', linewidth=0.5)
    ax.bar(np.arange(0, len(feature_types), 1) + 0.2, nonlinear_val_accs, width=0.4, label='nonlinear', color=COLORS[1], edgecolor='k', linewidth=0.5)

    # Custom legend. Colors indicate annotation-free or annotation-full, and hatching indicates training or validation set. 
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[0], edgecolor='k', linewidth=0.5)]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=COLORS[1], edgecolor='k', linewidth=0.5))
    plt.legend(handles, ['logistic', 'nonlinear'], ncol=2, fontsize=7, columnspacing=0.3, handletextpad=0.3, loc='upper left', bbox_to_anchor=(0.25, 0.99))

    _format_barplot_axes(ax, feature_types=feature_types, binary=binary)

    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=500, format='PNG', bbox_inches='tight')
        plt.close()  # Prevent figure from being displayed in notebook.
    else:
        plt.show()