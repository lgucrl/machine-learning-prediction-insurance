# Machine Learning prediction of insurance cross-selling 

This project builds a supervised **binary classification** model that helps an insurance company identify **cross-selling opportunities**: among existing customers, who is most likely to be interested in purchasing an additional **vehicle insurance** policy. The work follows an end-to-end Machine Learning pipeline (EDA, preprocessing, imbalance handling, model training, evaluation) and compares multiple algorithms and thresholds to balance decision strategies like identifying as many interested customers as possible (high recall) while controlling false positives (precision).

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
