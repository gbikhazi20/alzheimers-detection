import matplotlib.pyplot as plt
from collections import defaultdict
from eval import EvalResults
import utils
import numpy as np
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import os

# Disclaimer - The code for the visualizations is mostly AI generated
class Vizualizer:
    def __init__(self, results: EvalResults):
        self.preds = results.preds
        self.labels = results.labels
        self.outputs = results.outputs
    
    def make_plots(self):
        self.class_accuracies = self.__plot_class_accuracies()
        self.confusion_matrix = self.__plot_confusion_matrix()
        self.roc_curve = self.__plot_roc_curve()


    def __compute_per_class_accuracy(self):
        preds = self.preds
        labels = self.labels
        num_classes = len(utils.CLASSES)

        correct_per_class = defaultdict(int)  # Correct predictions per class
        total_per_class = defaultdict(int)  # Total instances per class

        for pred, label in zip(preds, labels):
            total_per_class[label] += 1
            if pred == label:
                correct_per_class[label] += 1

        per_class_accuracy = {class_idx: correct_per_class[class_idx] / total_per_class[class_idx]
                            for class_idx in range(num_classes)}

        return per_class_accuracy
    
    def __plot_class_accuracies(self):
        per_class_accuracy = self.__compute_per_class_accuracy()
        
        classes = utils.CLASSES
        accuracies = list(per_class_accuracy.values())

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(classes, accuracies, color='skyblue')
        ax.set_xlabel('Class')
        ax.set_ylabel('Accuracy')
        ax.set_title('Per-Class Accuracy')
        ax.set_xticks(classes)  # Ensure each class has a label on the x-axis
        ax.set_ylim(0, 1)  # Accuracy ranges from 0 to 1
        
        return fig

    def __plot_confusion_matrix(self):
        labels = np.array(self.labels)
        preds = np.array(self.preds)
        
        cm = confusion_matrix(labels, preds)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=utils.CLASSES, yticklabels=utils.CLASSES, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        
        return fig
    
    def __plot_roc_curve(self):
        probabilities = softmax(self.outputs, axis=1)

        fig, ax = plt.subplots(figsize=(10, 8))
        n_classes = probabilities.shape[1]
        
        colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            # Create binary labels for the current class
            binary_labels = np.array(np.array(self.labels) == i, dtype=np.int32)
            fpr, tpr, _ = roc_curve(binary_labels, probabilities[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f'ROC Class {i} (AUC = {roc_auc:.2f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curves')
        ax.legend(loc="lower right")
        
        return fig


def plot_class_accuracies(per_class_accuracy):
    # Create a bar plot for the per-class accuracy
    classes = utils.CLASSES
    accuracies = list(per_class_accuracy.values())

    plt.figure(figsize=(8, 6))
    plt.bar(classes, accuracies, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.xticks(classes)  # Ensure each class has a label on the x-axis
    plt.ylim(0, 1)  # Accuracy ranges from 0 to 1
    plt.show()



if __name__ == "__main__":

    for results_file in os.listdir("results"):
        results_path = os.path.join("results", results_file)
        results = utils.load_results(results_path)
        viz = Vizualizer(results)
        viz.make_plots()
        ca_plot = viz.class_accuracies
        cm_plot = viz.confusion_matrix
        roc_plot = viz.roc_curve
        ca_plot.savefig(f"visualizations/{results_file}_class_accuracies.png")
        cm_plot.savefig(f"visualizations/{results_file}_confusion_matrix.png")
        roc_plot.savefig(f"visualizations/{results_file}_roc_curve.png")
        print(f"Visualizations saved for {results_file}")