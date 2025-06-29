# Credit Risk Probability Model for Alternative Data

This project focuses on building an end-to-end machine learning solution to assess credit risk for a "buy-now-pay-later" service. Using transactional data from an e-commerce platform, we will engineer features, develop a predictive model to assign a risk probability score to customers, and set the foundation for a deployable API.

---

## 📂 Folder Structure

The project follows a standardized structure to ensure maintainability and scalability.

```
credit-risk-model/
├── .github/              # Contains GitHub Actions CI/CD workflows
├── data/
│   ├── raw/              # Raw, immutable data files (data.csv)
│   └── processed/        # Cleaned and processed data ready for modeling
├── notebooks/
│   └── 1.0-eda.ipynb     # Exploratory Data Analysis and initial hypotheses
├── src/
│   ├── api/              # Source code for the FastAPI application
│   ├── __init__.py
│   ├── data_processing.py# Scripts for feature engineering and data cleaning
│   ├── predict.py        # Script for generating predictions with a trained model
│   └── train.py          # Script for training the machine learning model
├── tests/                # Unit tests for the source code
├── .gitignore            # Specifies files and folders for Git to ignore
├── Dockerfile            # Instructions to build the Docker image for the API
├── docker-compose.yml    # Defines and runs the multi-container Docker application
├── README.md             # This file: project overview and instructions
└── requirements.txt      # Lists the Python packages required for the project
```

---

## 🚀 Setup and Usage

Follow these steps to set up your local environment and run the project.

### 1. Clone the Repository
```bash
git clone https://github.com/abeni505/week-5-credit-risk-mode.git
cd week-5-credit-risk-model
```

### 2. Create and Activate Virtual Environment
It is highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create the virtual environment
python3 -m venv venv

# Activate the environment
# On macOS and Linux:
source venv/bin/activate
# On Windows:
# .\venv\Scripts\activate
```

### 3. Install Dependencies
Install all the required Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Run Exploratory Data Analysis
The initial data exploration can be viewed and run using Jupyter Notebook.

```bash
# Make sure your virtual environment is active
jupyter notebook notebooks/1.0-eda.ipynb
```

# 📊 Credit Scoring Business Understanding

## 💡 How does the Basel II Accord's emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Accord fundamentally shapes our approach by requiring a model that is not only accurate but also **highly interpretable and thoroughly documented**. This is driven by its three-pillar structure:

### 🏛️ Pillar 1 (Minimum Capital Requirements)
Basel II allows banks to use internal models, like the one we are building, to calculate their Risk-Weighted Assets (RWA). A well-calibrated model can result in lower RWA and, consequently, **lower minimum capital requirements**, freeing up capital for the bank to invest or lend. However, the use of such internal models is a privilege granted only if the models are proven to be sound.

### 🏛️ Pillar 2 (Supervisory Review)
This is the most critical pillar influencing our work. Financial regulators have the authority to meticulously **review and validate a bank's internal risk models**. If a model is a "black box" (i.e., not interpretable), regulators cannot verify its logic, fairness, or stability over time. They can reject the model, forcing the bank to use a less risk-sensitive standardized approach, which typically results in higher capital charges. Therefore, **model interpretability is a non-negotiable prerequisite** for regulatory approval and operational use.

### 🏛️ Pillar 3 (Market Discipline)
This pillar mandates that banks **disclose their risk assessment processes and overall risk exposure** to the public and investors. This level of transparency is impossible without a model whose decisions can be clearly explained and justified. A well-documented, interpretable model is essential for fulfilling these disclosure requirements and maintaining market confidence.

---

## ❓ Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

### 🔶 Necessity of a Proxy Variable
Traditional credit risk models are trained on historical loan data that includes a clear, binary "default" label. Our **e-commerce dataset lacks this direct repayment information**, containing instead rich behavioral and transactional data. To build a supervised learning model that predicts risk, we must first engineer a target variable from this available data.

This engineered target is a **proxy variable**—an indirect measure that stands in for the true, unobserved default behavior. Based on the *"RFMS Method"* paper, we will use customer transaction patterns (Recency, Frequency, Monetary value) to identify a segment of disengaged users, who will serve as our proxy for "high-risk" customers.

### ⚠️ Business Risks of Using a Proxy

1. **False Positives (Rejecting Creditworthy Customers)**  
   The model might incorrectly flag a financially responsible customer as high-risk simply because they are an infrequent shopper on the e-commerce platform. This would result in **denying them credit**, leading to lost revenue opportunities and a poor customer experience.

2. **False Negatives (Accepting High-Risk Customers)**  
   The model might fail to identify a customer who is a genuine default risk because their transactional behavior (our proxy) does not reflect their underlying financial instability. Granting credit to these individuals would lead to **financial losses** when they inevitably default.

---

## 🔄 What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

### ✅ Logistic Regression (Simple & Interpretable)
- **Industry standard** for credit scoring.  
- Linear nature combined with **Weight of Evidence (WoE)** makes it highly transparent.  
- Each feature's contribution to the final credit score is **clear, additive, and easy to explain**.  
- Satisfies **Basel II's Pillar 2 requirements** for model transparency.  
- **Drawback:** Might not capture complex, non-linear patterns, potentially leading to slightly lower predictive accuracy.

### ⚡ Gradient Boosting (Complex & High-Performance)
- Very powerful with **higher accuracy** by identifying intricate data patterns.  
- **Opaque decision-making process** makes it nearly impossible to explain individual risk scores.  
- Lack of transparency is a major obstacle for **regulatory approval**, as adverse credit decisions must be explainable.

---

### 💡 **Conclusion**
In a regulated financial context, the need for **transparency, auditability, and fairness heavily outweighs** a marginal gain in predictive accuracy. The potential for **regulatory rejection** and inability to explain decisions make complex, black-box models like Gradient Boosting unsuitable for this core credit decisioning task.

✅ **An interpretable model like Logistic Regression is the strongly preferred and responsible choice.**

---

# Author
Abenezer M. Woldesenbet
