# 🧠 Employee Turnover Analytics

A data science and machine learning project to analyze employee behavior and predict attrition risk. This project helps HR departments identify patterns behind employee turnover and build predictive models for proactive decision-making.

---

## 📂 Project Overview

Employee attrition is a major challenge in many organizations. Using data-driven techniques, this project explores the key drivers of turnover and builds classification models to predict which employees are likely to leave.

### 💡 Objectives:
- Perform exploratory data analysis (EDA) on employee data
- Identify key features influencing turnover
- Build machine learning models to predict attrition
- Evaluate model performance using ROC-AUC and classification metrics

---

## 📊 Dataset

The dataset `HR_comma_sep.csv` (place in root folder) includes:

| Feature               | Description                                 |
|-----------------------|---------------------------------------------|
| satisfaction_level    | Employee satisfaction score                 |
| last_evaluation       | Most recent evaluation score                |
| number_project        | Total number of projects handled            |
| average_monthly_hours | Monthly average working hours               |
| time_spend_company    | Tenure in the company (years)               |
| work_accident         | 1 if the employee had a work accident       |
| promotion_last_5years | 1 if promoted in last 5 years               |
| department            | Department name                             |
| salary                | Categorical: low, medium, high              |
| left                  | Target variable (1 = Left, 0 = Stayed)      |

---

## ⚙️ Technologies & Libraries

- Python (Jupyter Notebook)
- pandas, numpy
- seaborn, matplotlib
- scikit-learn
- imbalanced-learn (SMOTE)



