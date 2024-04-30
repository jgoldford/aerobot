'''Code for generating figures from model outputs. Functions are designed to interface with results dictionaries, which are given as 
output by model training and evaluation scripts.'''
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np 
import pandas as pd
from aerobot.io import FEATURE_TYPES, FEATURE_SUBTYPES, RESULTS_PATH, load_results_dict
import matplotlib.ticker as ticker
import seaborn as sns
from typing import Dict, NoReturn, List
import os

# TODO: Work this in to the io.py module. Also add other feature types. 
PRETTY_NAMES = {'KO':'All gene families', 'embedding.geneset.oxygen':'Five-gene set', 'chemical':'Chemical features', 'aa_1mer':'Amino acid counts', 'aa_3mer':'Amino acid trimers'}
PRETTY_NAMES.update({'embedding.genome':'Genome embedding', 'metadata':'Metadata'})
PRETTY_NAMES.update({f'nt_{i}mer':f'Nucleotide {i}-mer' for i in range(1, 6)})
PRETTY_NAMES.update({f'cds_{i}mer':f'CDS {i}-mer' for i in range(1, 6)})
PRETTY_NAMES.update({f'aa_{i}mer':f'Amino acid {i}-mer' for i in range(2, 3)})


CMAP = mpl.colormaps['GnBu']
COLORS = ['tab:gray', 'tab:green', 'tab:blue', 'tab:olive', 'tab:cyan', 'tab:brown']

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
    fig, ax = plt.subplots(1, figsize=(9, 3))

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

    fig, ax = plt.subplots(1, figsize=(9, 3))

    feature_types = list(nonlinear_results.keys())
    for feature_type in feature_types:
        assert feature_type in logistic_results, f'plot_nonlinear_logistic_comparison_barplot: {feature_type} is missing in the logistic regression results.'
    
    # Extract the final balanced accuracies on from the results dictionaries. 
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


def plot_confusion_matrices(results:Dict[str, Dict], path:str=None) -> NoReturn:

    feature_types = list(results.keys())
    binary = results[feature_types[0]]['binary'] # Assume all have the same value, but might want to add a check.
    classes = results[feature_types[0]]['classes'] # This should also be the same for each feature type. 

    # Extract the confusion matrices, which are flattened lists and need to be reshaped. 
    dim = 2 if binary else 3 # Dimension of the confusion matrix.
    confusion_matrices = [results[feature_type]['confusion_matrix'] for feature_type in feature_types]
    confusion_matrices = [np.array(confusion_matrix).reshape(dim, dim) for confusion_matrix in confusion_matrices]
    
    # Initialize a figure with one axis for each feature type. These will all share a y axis.
    fig, axes = plt.subplots(ncols=len(feature_types), figsize=(10, 3), sharey=True, sharex=False)
    axes = axes.flatten()
    plt.ylabel('true label') # Only need to share once if they share a y-axis.

    for feature_type, confusion_matrix, ax in zip(feature_types, confusion_matrices, axes):
        confusion_matrix = pd.DataFrame(confusion_matrix, columns=classes, index=classes)
        confusion_matrix = confusion_matrix.apply(lambda x: x/x.sum(), axis=1) # Normalize the matrix.
        ax.set_xlabel('predicted label')
        ax.set_title(PRETTY_NAMES[feature_type], loc='center')
        sns.heatmap(confusion_matrix, ax=ax, cmap='Blues', annot=True, fmt='.1%', cbar=False)
        # Rotate the tick labels on the x-axis of each subplot.
        ax.set_xticks(np.arange(len(classes)) + 0.5, classes, rotation=45)
    
    axes[0].set_yticks(np.arange(len(classes)) + 0.5, classes, rotation=0)

    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=500, format='PNG', bbox_inches='tight')
        plt.close()  # Prevent figure from being displayed in notebook.
    else:
        plt.show()


def plot_phylo_bias(results:Dict[str, Dict], show_points:bool=False, path:str=None) -> NoReturn:
    '''Plots the results of a single run of phlogenetic bias analysis''' 

    feature_type = results['feature_type'] # Extract the feature type from the results dictionary. 
    levels = ['Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'][::-1]
    fig, ax = plt.subplots(figsize=(5, 3))

    # Load in results for the baselines. 
    randrel_results, meanrel_results = None, None
    if 'phylo_bias_results_randrel.json' in os.listdir(RESULTS_PATH):
        randrel_results = load_results_dict(os.path.join(RESULTS_PATH, 'phylo_bias_results_randrel.json'))
    if 'phylo_bias_results_meanrel.json' in os.listdir(RESULTS_PATH):
        meanrel_results = load_results_dict(os.path.join(RESULTS_PATH, 'phylo_bias_results_meanrel.json'))

    colors = ['gray', 'black', 'tab:blue']
    linestyles = ['--', '--', '-']
    labels = ['MeanRelative', 'RandomRelative', None if results is None else results['model_class'].capitalize()]
    legend = []

    for i, results in enumerate([meanrel_results, randrel_results, results]):
        if results is not None:
            # Plot the error bar, as well as scatter points for each level. 
            means = [results['scores'][level]['mean'] for level in levels] # Extract the mean F1 scores.
            errs = [results['scores'][level]['err'] for level in levels] # Extract the standard errors. 
            level_scores = [results['scores'][level]['scores'] for level in levels] # Extract the raw scores for each level. 
            # Convert the scores to points for a scatter plot. 
            scores_x = np.ravel([np.repeat(i + 1, len(s)) for i, s in enumerate(level_scores)])
            scores_y = np.ravel(level_scores)

            ax.errorbar(np.arange(1, len(levels) + 1), means, yerr=errs, c=colors[i], linestyle=linestyles[i], capsize=3)
        
            if show_points: # Only show the points if specified.
                ax.scatter(scores_x, scores_y, color=colors[i], s=3)
            
            legend.append(labels[i])

    ax.set_ylabel('balanced accuracy')
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(1, len(levels) + 1), labels=levels)
    ax.set_xlabel('holdout level')
    ax.set_title(f'Phylogenetic bias analysis for {PRETTY_NAMES[feature_type]}')
    ax.legend(legend)

    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=500, format='PNG', bbox_inches='tight')
        plt.close()  # Prevent figure from being displayed in notebook.
    else:
        plt.show()

# def plot_phylo_bias(
#     nonlinear_results:Dict[str, Dict[str, Dict]]=None, 
#     logistic_results:Dict[str, Dict[str, Dict]]=None, 
#     meanrel_results:Dict[str, Dict]=None,
#     path:str=None, 
#     show_points:bool=False) -> NoReturn:
    
#     levels = ['Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'][::-1]
#     fig, ax = plt.subplots(figsize=(15, 6))
#     # Get a set of all feature types present in both input dictionaries. 
#     colors = CMAP(np.linspace(0.2, 1, len(FEATURE_TYPES)))
#     legend = []

#     def _plot(results:Dict, color:str=None, linestyle='-'):
#             # Plot the error bar, as well as scatter points for each level. 
#             means = [results['scores'][level]['mean'] for level in levels] # Extract the mean F1 scores.
#             errs = [results['scores'][level]['err'] for level in levels] # Extract the standard errors. 
#             level_scores = [results['scores'][level]['scores'] for level in levels] # Extract the raw scores for each level. 
#             # Convert the scores to points for a scatter plot. 
#             scores_x = np.ravel([np.repeat(i + 1, len(s)) for i, s in enumerate(level_scores)])
#             scores_y = np.ravel(level_scores)
#             ax.errorbar(np.arange(1, len(levels) + 1), means, yerr=errs, c=color, capsize=3)

#             if show_points: # Only show the points if specified.
#                 ax.scatter(scores_x, scores_y, color=color)

#     for feature_type, color in zip(FEATURE_TYPES, colors):
#         if (nonlinear_results is not None) and (feature_type in nonlinear_results):
#             results = nonlinear_results[feature_type]
#             _plot(results, color=color, linestyle='-')
#             legend.append(f'{PRETTY_NAMES[feature_type]} (nonlinear)')
        
#         if (logistic_results is not None) and (feature_type in logistic_results):
#             results = logistic_results[feature_type]
#             _plot(results, color=color, linestyle='--')
#             legend.append(f'{PRETTY_NAMES[feature_type]} (logistic)')

#     if (meanrel_results is not None):
#         _plot(meanrel_results, color='black', linestyle='--')

#     ax.legend(legend, bbox_to_anchor=(1.3, 1))
#     ax.set_ylabel('balanced accuracy')
#     ax.set_ylim(0, 1)
#     ax.set_xticks(np.arange(1, len(levels) + 1), labels=levels)
#     ax.set_xlabel('holdout level')
#     ax.set_title(f'Phylogenetic bias analysis')

#     plt.tight_layout()
#     if path is not None:
#         plt.savefig(path, dpi=500, format='PNG', bbox_inches='tight')
#         plt.close()  # Prevent figure from being displayed in notebook.
#     else:
#         plt.show()


