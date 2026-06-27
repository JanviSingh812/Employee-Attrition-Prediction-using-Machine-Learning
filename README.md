

---

## 💼 Project Overview
Employee attrition represents a significant cost and operational challenge to modern corporations. This project aims to diagnose historical flight risk patterns and build a predictive classifier to identify at-risk employees. By analyzing corporate workforce demographics, compensation, job roles, and work-life balance, we isolate the main drivers of employee turnover.

---

## 📊 Dataset & Preprocessing
The dataset used is **[HR_Attrition.csv](HR_Attrition.csv.csv)** containing **1,470 employee records and 35 features**.

### Preprocessing Steps:
1. **Data Cleaning:** Verified that the dataset contains **0 missing values**.
2. **Feature Pruning:** Removed non-predictive administrative identifiers and constant columns:
   - `EmployeeCount`, `EmployeeNumber`, `Over18`, `StandardHours`
3. **Target Mapping:** Converted the text-based `Attrition` column into a binary target (`Yes` -> `1`, `No` -> `0`).
4. **Class Imbalance Handling:** The dataset is highly imbalanced with an attrition rate of **16.12%** (1,233 Stayed vs. 237 Left). We handled this by applying class balancing (`class_weight='balanced'`) to prevent bias toward the majority class.
5. **Categorical Encoding:** Applied one-hot encoding to all categorical variables.
6. **Feature Scaling:** Applied `StandardScaler` to align features on a uniform scale.
7. **Data Partitioning:** Utilized a stratified 80/20 train/test split to preserve the target ratio in both sets.

---

## 💡 Key Business Insights (EDA)
Exploratory analysis of the corporate dataset revealed several critical vulnerabilities:
* **Department Vulnerability:** The **Sales Department** experiences the highest turnover rate at **20.63%**, closely followed by **Human Resources** at **19.05%**. Research & Development is the most stable at **13.84%**.
* **High-Risk Job Roles:** **Sales Representatives** are at extreme risk of turnover, showing an alarming attrition rate of **39.76%**. **Laboratory Technicians** also show elevated attrition at **23.94%**.
* **Compensation Anchor:** Income acts as a significant retaining factor. Employees who stay have an average monthly income of **$6,832.74**, whereas those who choose to exit have a significantly lower average monthly income of **$4,787.09**.
* **Work-Life Balance:** Employees scoring their work-life balance at the lowest tier (**Level 1**) experience a massive attrition rate of **31.25%**, which is double the rate seen in higher tiers.
* **Tenure Burnout Window:** Employees who resign do so relatively early in their tenure, with an average of **5.13 years** at the company, compared to a stable **7.37 years** for retained staff.

---

## 🧠 Model Comparison & Evaluation
We evaluated three different machine learning models using a stratified validation set. 

| Model | Precision | Recall | F1-Score | ROC-AUC Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced)** | 0.3452 | **0.6170** | **0.4427** | **0.7983** |
| **Gradient Boosting (Weighted)** | 0.4074 | 0.4681 | 0.4356 | 0.7791 |
| **Random Forest (Balanced)** | **0.4444** | 0.0851 | 0.1429 | 0.7547 |

### Why Logistic Regression is the Selected Model:
In corporate retention models, **Recall** is the most critical metric. We want to correctly identify the highest proportion of actual flight-risk employees, even at the cost of a few false positives. **Logistic Regression** maximizes Recall (**61.70%**) and yields the highest overall ROC-AUC score (**0.7983**). Additionally, its linear coefficients provide perfect explainability, allowing HR teams to understand *why* the model predicts flight risk for specific individuals.

---

## 📈 Top 10 Drivers of Attrition
Analysis of the Logistic Regression coefficients isolates the top ten variables contributing to employee resignation (ranked by absolute coefficient value):

1. **JobRole: Laboratory Technician** (+0.810) - High exit likelihood.
2. **OverTime: Yes** (+0.771) - Strong predictor of burnout.
3. **BusinessTravel: Travel Frequently** (+0.723) - Travel stress.
4. **TotalWorkingYears** (-0.660) - Retentive factor; employees with more overall work experience are more stable.
5. **JobLevel** (-0.650) - Retentive factor; higher career levels correlate with lower attrition.
6. **JobRole: Sales Representative** (+0.531) - High exit likelihood.
7. **BusinessTravel: Travel Rarely** (+0.513) - Positive coefficient relative to no travel.
8. **EducationField: Life Sciences** (+0.512) - Relative flight risk compared to other fields.
9. **YearsSinceLastPromotion** (+0.499) - Career stagnation increases exit likelihood.
10. **Department: Sales** (+0.471) - Structural attrition risk.

---

## 📁 Repository Structure
```
├── HR_Attrition.csv.csv     # Raw corporate dataset
├── analysis.ipynb           # Jupyter Notebook containing EDA, preprocessing, and model pipelines
├── summary.pdf.pdf          # Detailed PDF reporting final project findings
├── README.md                # Project documentation
└── charts/                  # High-resolution visual dashboards
    ├── chart1_attrition_by_role.png      # Attrition rate across departments and roles
    ├── chart2_income_distribution.png    # Monthly income density vs attrition
    ├── chart3_confusion_matrix.png       # Confusion matrix of the chosen Logistic Regression model
    ├── chart4_feature_importance.png     # Sorted absolute coefficient values from the model
    └── chart5_comparative_roc.png        # Comparative ROC curves for the models
```

---

## 🚀 How to Run the Analysis
1. Clone this repository:
   ```bash
   git clone https://github.com/JanviSingh812/-Employee-Attrition-Prediction-using-Machine-Learning.git
   ```
2. Navigate to the project directory:
   ```bash
   cd -Employee-Attrition-Prediction-using-Machine-Learning
   ```
3. Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn notebook
   ```
4. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
5. Open **[analysis.ipynb](analysis.ipynb)** and run all cells to reproduce the exploratory findings, model training, evaluation, and visualizations.
