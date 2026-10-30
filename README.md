# Towards Accurate Brain Stroke Prediction

A machine learning-based project that analyzes patient demographic,
clinical, and lifestyle information to predict the likelihood of brain
stroke. Multiple classification algorithms are trained and evaluated
using performance metrics and confusion matrices to compare their
predictive capabilities.

## Project Overview

Brain stroke is a major health concern that can lead to long-term
disability or death. Early identification of individuals who may be at
risk can support timely medical assessment and preventive care.

This project explores the use of machine learning for binary stroke
prediction. It follows a workflow involving dataset preparation,
exploratory data analysis, preprocessing, feature selection, model
training, and performance evaluation.

The project report is titled **"Towards Accurate Brain Stroke
Prediction: Integrating Clinical and Genetic Data."** The implementation
documented in the report uses a dataset containing 11 demographic,
medical, and lifestyle features. Genetic variables are discussed as part
of the broader research motivation and future direction; the reported
implementation does not document an experimental genetic-data
integration pipeline.

## Objectives

-   Explore patient-related factors associated with stroke occurrence.
-   Prepare clinical and demographic data for machine learning.
-   Train and compare multiple classification algorithms.
-   Evaluate models using accuracy, precision, recall, F1-score,
    confusion matrices, and ROC curves.
-   Identify promising models for further research into stroke-risk
    assessment.

## Technologies Used

  Technology     Purpose
  -------------- -------------------------------------------------
  Python         Programming language
  NumPy          Numerical operations
  Pandas         Dataset loading and manipulation
  Scikit-learn   Preprocessing, model training, and evaluation
  XGBoost        Gradient-boosting classification
  Matplotlib     Data and result visualization
  Seaborn        Statistical plots and confusion-matrix heatmaps
  Anaconda       Python environment and package management

## Dataset

The report describes a brain stroke dataset with 11 input/target fields:

  Feature               Description
  --------------------- -------------------------------------------
  `gender`              Patient gender
  `age`                 Patient age in years
  `hypertension`        Hypertension indicator (1 = Yes, 0 = No)
  `heart_disease`       Heart disease indicator (1 = Yes, 0 = No)
  `ever_married`        Marital status
  `work_type`           Employment category
  `Residence_type`      Urban or rural residence
  `avg_glucose_level`   Average glucose level
  `bmi`                 Body Mass Index
  `smoking_status`      Smoking category
  `stroke`              Target (1 = Stroke, 0 = No Stroke)

The report describes the dataset as imbalanced, with approximately 5%
stroke cases and 95% non-stroke cases. It identifies Kaggle as a source
for publicly available data, but does not provide a definitive dataset
URL in the implementation section. Use the exact dataset file associated
with your project and comply with its license and usage conditions.

### Expected dataset file

The implementation example in the report loads a CSV named
`brain_stroke.csv`. A suggested repository layout is:

``` text
project/
├── data/
│   └── brain_stroke.csv
├── notebooks/
│   └── stroke_prediction.ipynb
├── README.md
└── requirements.txt
```

The notebook and folder names above are suggested organization; adapt
them to match the files actually present in your repository. Do not
upload private or identifiable patient information.

## Methodology

The project workflow consists of the following stages:

1.  **Data collection:** Load the brain stroke dataset.
2.  **Data preprocessing:** Prepare categorical and numerical features
    for model training.
3.  **Exploratory data analysis:** Examine feature distributions and
    relationships using visualizations such as heatmaps, scatter plots,
    and bar plots.
4.  **Feature selection:** Use correlation analysis and, where
    applicable, Recursive Feature Elimination (RFE) to examine relevant
    variables.
5.  **Model training:** Train multiple supervised classification
    algorithms.
6.  **Model evaluation:** Compare predictions using classification
    metrics, confusion matrices, and ROC curves.
7.  **Model comparison:** Identify promising models for further testing
    and improvement.

### Data preprocessing

The report describes these preprocessing operations:

-   **Missing values:** Fill missing BMI values using median imputation.
-   **Categorical encoding:** Apply Label Encoding to categorical fields
    such as gender, work type, smoking status, and residence type.
-   **Feature scaling:** Apply Min-Max normalization to continuous
    variables such as age, average glucose level, and BMI.
    Standardization is also mentioned in the model-building discussion.
-   **Class imbalance:** Discuss the use of SMOTE oversampling and/or
    class weights to address the low proportion of stroke cases.
-   **Data splitting:** Use an 80:20 training/testing split.
-   **Cross-validation:** Use 5-fold cross-validation on training data,
    as described in the proposed methodology.

The report contains descriptions of alternative preprocessing choices in
different sections. The exact preprocessing sequence should therefore be
confirmed against the final source code before reproducing the
experiments.

## Machine Learning Algorithms

The project report evaluates or discusses the following classifiers:

-   **Logistic Regression:** An interpretable baseline classifier for
    estimating stroke probability.
-   **Gaussian Naive Bayes:** A probabilistic classifier based on
    conditional feature-independence assumptions.
-   **Support Vector Machine (SVM):** A classifier that separates
    classes using a decision boundary; a linear SVM is described in the
    methodology.
-   **K-Nearest Neighbors (KNN):** Classifies a patient based on nearby
    training examples.
-   **Decision Tree:** Uses feature-based decision rules and provides
    interpretable classification paths.
-   **Random Forest:** Combines predictions from multiple decision
    trees.
-   **XGBoost:** Uses gradient boosting to learn patterns in structured
    data.
-   **Neural Network (MLP):** A feedforward neural network is discussed
    in the proposed system.

## Reported Results

The following values are transcribed from **Table 1 (Recorded Results
for each Classifier)** in the project report.

  Model                      Accuracy   Precision   Recall   F1-score
  ------------------------ ---------- ----------- -------- ----------
  Logistic Regression           0.950       0.950    0.950      0.950
  Naive Bayes                   0.870       0.871    0.870      0.860
  Support Vector Machine        0.956       0.953    0.956      0.953
  Random Forest                 0.956       0.956    0.956      0.956
  Decision Tree                 0.920       0.950    0.960      0.960
  XGBoost                       0.980       0.981    0.980      0.980
  KNN                           0.940       0.950    0.970      0.990

According to the report, XGBoost recorded the highest accuracy (0.98),
while Random Forest and SVM also showed strong reported performance.

**Results note:** These are the values printed in the report, not
independently reproduced results. The KNN F1-score and some other metric
combinations should be checked against the original experiment outputs
before being treated as validated results. In particular, the report's
confusion matrices and summary table should be reconciled before making
model-selection claims.

## Evaluation Metrics

-   **Accuracy:** Overall proportion of correctly classified records.
-   **Precision:** Proportion of predicted stroke cases that are actual
    stroke cases.
-   **Recall (Sensitivity):** Proportion of actual stroke cases
    correctly identified.
-   **F1-score:** Harmonic mean of precision and recall.
-   **Confusion matrix:** Shows true positives, true negatives, false
    positives, and false negatives.
-   **ROC curve / ROC-AUC:** Examines the trade-off between
    true-positive and false-positive rates across classification
    thresholds.

Because the dataset is imbalanced, accuracy alone is not sufficient to
judge a model. Recall and the confusion matrix are particularly
important when examining missed stroke cases.

## Environment Setup

The report describes using Python and Anaconda, with a desktop
environment of at least 8 GB RAM and an Intel i5 processor.

Create and activate a virtual environment:

``` bash
python -m venv .venv
```

Activate it:

**Windows**

``` bash
.venv\Scripts\activate
```

**macOS / Linux**

``` bash
source .venv/bin/activate
```

Install the main libraries:

``` bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

If your repository includes a `requirements.txt`, install from it
instead:

``` bash
pip install -r requirements.txt
```

## Running the Project

1.  Clone the repository:

    ``` bash
    git clone <YOUR_REPOSITORY_URL>
    ```

2.  Move into the project directory:

    ``` bash
    cd <YOUR_PROJECT_FOLDER>
    ```

3.  Create and activate the Python environment using the commands above.

4.  Install the required libraries.

5.  Place the project dataset at the path expected by your notebook or
    Python script. The sample implementation in the report uses:

    ``` text
    brain_stroke.csv
    ```

6.  Open Jupyter Notebook:

    ``` bash
    jupyter notebook
    ```

7.  Open and run the project's notebook or Python implementation.

> The report includes sample training code that reads
> `/content/brain_stroke.csv`, a path commonly used in hosted notebook
> environments. Update the data path to match your local repository
> structure before running it.

## Project Outputs

The report documents the following outputs:

-   Confusion matrices for the evaluated classifiers.
-   A comparative table of accuracy, precision, recall, and F1-score.
-   ROC curves for model comparison.
-   A discussion of model strengths, weaknesses, and classification
    errors.

## Limitations

-   The dataset is imbalanced, which can affect minority-class
    predictions.
-   Model performance depends on the quality, completeness, and
    representativeness of the dataset.
-   The report does not document a completed genetic-data integration
    experiment, despite the broader project title.
-   The reported metrics should be validated against the original code
    and experiment outputs.
-   Performance on one dataset does not establish generalization to
    other populations or clinical settings.
-   The project is an academic machine learning study and is not a
    clinically validated diagnostic system.

## Future Scope

Potential directions identified in the report include:

-   Integrating genomic, clinical, and demographic information in a
    clearly defined multimodal dataset.
-   Validating models using diverse populations and real-world data,
    such as patient registries and electronic health records.
-   Improving interpretability and transparency of model predictions.
-   Investigating feature engineering and hyperparameter optimization.
-   Monitoring model performance and potential bias as new data becomes
    available.
-   Exploring a web-based decision-support interface for research and
    demonstration.

## Disclaimer

This project is intended for academic and research purposes only. It
must not be used as a substitute for professional medical advice,
diagnosis, or treatment. A machine learning prediction cannot confirm or
rule out a stroke. Anyone experiencing possible stroke symptoms should
seek emergency medical care immediately.

## References

The project report includes a literature survey and references covering
stroke prediction, machine learning, stroke risk factors, and related
research. Refer to the original project report for the complete
bibliography.
