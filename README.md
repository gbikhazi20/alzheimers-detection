# alzheimers-detection

This library contains the code for our Alzheimer's detection final project.

The bulk of the code in this repo dedicated to model training lived in a Colab Notebook for most of this project's life. Much of it was written by [George Chakhnashvili](https://www.github.com/ShellRox), and some by [George Bikhazi](https://www.github.com/gbikhazi20), who refactored everything into this repo and added evaluation and visualization functionality.

We provide tools for:

- Training models on the [OASIS dataset](https://www.kaggle.com/datasets/ninadaithal/imagesoasis)
- Evaluating those models and creating visualizations from the results

The evaluation code assumes a `models` directory with state dictionaries of trained models.

Here are some example visualizations:

**Resnet confusion matrix**
<img src="visualizations/resnet.pkl_confusion_matrix.png" style="max-width:80%">

**Vision Transformer ROC Curves**
<img src="visualizations/vit.pkl_roc_curve.png" style="max-width:80%">

**Vision Transformer Class Accuracies**
<img src="visualizations/vit.pkl_class_accuracies.png" style="max-width:80%">

**And some preliminary experimentation with visualizing network activations using `activations.ipynb`**
<img src="visualizations/activations.png" style="max-width:80%">
