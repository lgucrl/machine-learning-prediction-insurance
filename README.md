# Machine Learning prediction of insurance cross-selling 

This project builds a supervised **binary classification** model in Python that helps an insurance company identify **cross-selling opportunities**: among existing customers, who is most likely to be interested in purchasing an additional **vehicle insurance** policy. The work follows an end-to-end Machine Learning pipeline (EDA, preprocessing, imbalance handling, model training, evaluation) and compares multiple algorithms and thresholds to balance decision strategies like identifying as many interested customers as possible (high recall) while controlling false positives (precision). The full project can be found in the [`insurance_ml_prediction.ipynb`](https://github.com/lgucrl/machine-learning-cross-selling-prediction/blob/main/insurance_ml_prediction.ipynb) notebook. 

---

## Dataset

The `insurance_cross_sell.csv` dataset contains detailed information about customers and their historical/behavioral attributes. It includes the following variables:

- **Gender**: gender of the customer.
- **Age**: age of the customer.
- **Driving_License**:
  - `0`: customer doesn't have a driving license  
  - `1`: customer already has a driving license
- **Region_Code**: unique code for the customer’s region.
- **Previously_Insured**:
  - `0`: customer doesn't have vehicle insurance  
  - `1`: customer already has vehicle insurance
- **Vehicle_Age**: age category of the vehicle (e.g., `< 1 Year`, `1-2 Year`, `> 2 Years`).
- **Vehicle_Damage**:
  - `0`: customer didn’t have vehicle damage in the past  
  - `1`: customer had vehicle damage in the past
- **Annual_Premium**: amount the customer pays as premium per year.
- **Policy_Sales_Channel**: code describing how the customer was approached (agent, phone, mail, in person, etc.).
- **Vintage**: number of days the customer has been associated with the company.
- **Response (target variable)**:
  - `0`: customer is not interested in the company’s vehicle insurance  
  - `1`: customer is interested in the company’s vehicle insurance

---

## Project workflow

1. **Exploratory Data Analysis (EDA)**  
   The project begins by validating the dataset structure (size, data types, missing values) and exploring feature distributions. Numeric variables like `Age`, `Annual_Premium`, and `Vintage` are checked for skewness and outliers, while categorical fields like `Gender`, `Vehicle_Age`, and `Vehicle_Damage` are analyzed for class proportions and their relationship to `Response`, to provide intuition about which features may be predictive.

2. **Feature preparation and encoding**  
   Raw variables are transformed into a clean feature matrix suitable for classical ML models. Binary categories are mapped to 0/1, while multi-class features (e.g., `Vehicle_Age`, `Region_Code`, `Policy_Sales_Channel`) are encoded using **one-hot encoding**. Non-informative identifiers (e.g., customer id) are excluded to reduce noise and prevent leakage.

3. **Train/test split and scaling**  
   Data is split into training and test sets using a **stratified split** to preserve the original class distribution in both subsets. Feature **standardization** is applied on the training set only and transforming the test set afterward to ensure fair comparisons across algorithms and avoid data leakage.

4. **Handling class imbalance with resampling**  
   To address imbalance, multiple training set variants are created and compared. The workflow evaluates several **undersampling** ratios (reducing the majority class) and a **combined approach** that oversamples the minority class and then undersamples to a balanced distribution. The test set is never resampled, preserving a realistic evaluation scenario.

5. **Model training and comparison**  
   Several classification models are trained, including **Logistic Regression**, **K-Nearest Neighbors (KNN)**, and **Random Forest**. Each model is fitted across the different resampled training sets to observe how algorithm choice and class balancing impact performance.

6. **Evaluation and threshold tuning**  
   Models are assessed using **confusion matrices** and core classification metrics (**precision**, **recall**, **F1**, **PR-AUC**) that better reflect performances where the positive class is typically the minority, wihout relying on accuracy alone. The project also explores **threshold tuning**: adjust the decision threshold (e.g., from 0.5 to a higher value) to reduce false positives when the goal is to identify only the most promising customers, or keep it lower to maximize recall when the priority is capturing as many interested customers as possible.
