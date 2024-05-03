
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.metrics import (auc, f1_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

from ..utils.tcga_post_analysis_utils import Results_Handler


def compute_prc(test_labels,lr_probs,yhat,results:Results_Handler,show_figure:bool=False):

    lr_precision, lr_recall, _ = precision_recall_curve(test_labels, lr_probs)
    lr_f1, lr_auc = f1_score(test_labels, yhat), auc(lr_recall, lr_precision)
    # plot the precision-recall curves
    if show_figure:
        pyplot.plot(lr_recall, lr_precision, marker='.', label=results.figures_path.split('/')[-2])
        # axis labels
        pyplot.xlabel('Recall')
        pyplot.ylabel('Precision')
        # show the legend
        pyplot.legend()
    # save and show the plot
    plt.title("PRC Curve")

    if results.save_results:
        plt.savefig(f"{results.figures_path}/prc_curve.png")
        plt.savefig(f"{results.figures_path}/prc_curve.svg")

    if show_figure:
        plt.show()
    return lr_f1,lr_auc

def compute_roc(test_labels,lr_probs,results,show_figure:bool=False):
    
    ns_probs = [0 for _ in range(len(test_labels))]

    # calculate scores
    ns_auc = roc_auc_score(test_labels, ns_probs)
    lr_auc = roc_auc_score(test_labels, lr_probs)
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(test_labels, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(test_labels, lr_probs)

    # plot the roc curve for the model
    if show_figure:
        plt.plot(lr_fpr, lr_tpr, marker='.',markersize=1, label=results.figures_path.split('/')[-2])
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
        plt.title("ROC Curve")

    if results.save_results:
        plt.savefig(f"{results.figures_path}/roc_curve.png")
        plt.savefig(f"{results.figures_path}/roc_curve.svg")

    if show_figure:    
        plt.show()
    return lr_auc
