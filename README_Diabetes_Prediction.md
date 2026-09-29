# Diabetes Prediction

A machine learning classification project that predicts whether a person is diabetic (`1`) or non-diabetic (`0`) using medical and demographic features from the diabetes dataset.

## Dataset

The notebook loads `diabetes.csv`.

- **Rows:** 768
- **Columns:** 9
- **Features:** 8
- **Target:** `Outcome`

### Features

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Glucose level |
| BloodPressure | Blood pressure |
| SkinThickness | Skin thickness |
| Insulin | Insulin level |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age |

`Outcome` is the target:
- `0` = non-diabetic
- `1` = diabetic

The dataset contains 500 samples with outcome `0` and 268 samples with outcome `1`.

## Exploratory Analysis in the Notebook

The notebook checks:

- First and last rows of the dataset
- Dataset shape
- Descriptive statistics using `describe()`
- Target class distribution using `value_counts()`
- Mean feature values grouped by `Outcome`

The grouped means show differences between the two outcome groups. For example, the mean glucose value is about **109.98** for outcome `0` and **141.26** for outcome `1`.

## Preprocessing

The notebook follows these steps:

1. Separates the features (`X`) from the target (`Y`).
2. Applies `StandardScaler` to the feature data.
3. Uses the standardized data for model training.
4. Splits the data into training and testing sets using:
   - `test_size=0.2`
   - `stratify=Y`
   - `random_state=2`

The resulting split is:

- Training set: **614 samples**
- Test set: **154 samples**

## Model

The notebook uses a **Support Vector Machine (SVM)** classifier:

```python
SVC(kernel='linear')
```

The model is trained on the standardized training data.

## Results

The notebook reports the following accuracy scores:

| Dataset | Accuracy |
|---|---:|
| Training | **78.66%** |
| Test | **77.27%** |

The test accuracy is calculated using the predictions from the held-out test set.

## Prediction

The notebook starts a predictive-system section at the end, but the uploaded notebook does not contain a completed input example or final prediction output.

## Libraries Used

- NumPy
- Pandas
- Scikit-learn

Main Scikit-learn components used:

- `train_test_split`
- `LogisticRegression` (imported but not used in the shown model)
- `StandardScaler`
- `SVC`
- `accuracy_score`

## Project Workflow

```text
Diabetes Dataset
       ↓
Data Inspection
       ↓
Separate Features & Target
       ↓
StandardScaler
       ↓
Train-Test Split
       ↓
Linear SVM Classifier
       ↓
Model Evaluation
       ↓
Accuracy Score
```

## How to Run

1. Place `diabetes.csv` in the project directory.
2. Open the notebook in Jupyter Notebook or JupyterLab.
3. Update the dataset path in the `pd.read_csv()` cell if required.
4. Run the cells from top to bottom.

## Notes

This README reflects the implementation and results present in the uploaded notebook. The notebook uses accuracy as its reported evaluation metric and does not include a confusion matrix, classification report, ROC-AUC score, or completed prediction example.
