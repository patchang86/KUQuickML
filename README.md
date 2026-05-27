KUQuickML
KUQuickML is a graphical user interface-based machine learning platform designed to improve accessibility for beginners and researchers in plant biotechnology with limited programming experience.
Features
CSV-based data loading
Feature/label role assignment
Data preprocessing and scaling
Train-test splitting with optional stratification
Supervised learning with:
K-nearest neighbors (KNN)
Support vector machine (SVM)
Multilayer perceptron (MLP)
Random forest (RF)
Model comparison
5-fold cross-validation
Hyperparameter tuning with Optuna
Confusion matrix and feature-ranking outputs
Two-dimensional visualization using PCA, LDA, and NCA
Model save/load using `joblib`
Log extraction in `.txt` format
Supported tasks
Classification
Regression
Installation
Option 1. From source
Clone the repository:
```bash
   git clone https://github.com/patchang86/KUQuickML.git
   cd KUQuickML
   ```
Create and activate a virtual environment:
```bash
   python -m venv .venv
   ```
On Windows:
```bash
   .venv\Scripts\activate
   ```
On macOS/Linux:
```bash
   source .venv/bin/activate
   ```
Install dependencies:
```bash
   pip install -r requirements.txt
   ```
Run KUQuickML:
```bash
   python main.py
   ```
Example workflow
Load a CSV dataset.
Assign columns as sample identifiers, labels, or features.
Apply an optional scaler.
Split the dataset into training and test sets.
Train a model or compare multiple models.
Evaluate results using built-in metrics and visual outputs.
Save the trained model and export the log file.
Example datasets
The manuscript validates KUQuickML using:
Iris dataset
One-hundred plant species leaves dataset
Arabidopsis thaliana metabolomics dataset
Platform note
KUQuickML was developed primarily in a Windows environment. macOS and Linux compatibility is intended, but testing on those platforms has been more limited, and some functions may not operate identically across systems.
## Citation

If you use KUQuickML in academic work, please cite the associated manuscript and the archived software release.

Suggested software citation:

> Do E, Ku K-M. KUQuickML (Version 1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.20416009
>
> ## License

KUQuickML is distributed under the MIT License. See the `LICENSE` file for details.

The source code repository is also available at: https://github.com/patchang86/KUQuickML
