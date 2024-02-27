import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Support functions
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform

# Fit models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Scoring functions
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

import shap

# Setting display options for better readability
pd.options.display.max_rows = None
pd.options.display.max_columns = None

optimize = False

# Function to give best model score and parameters
def best_model(model):
    print(model.best_score_)
    print(model.best_params_)
    print(model.best_estimator_)


def get_auc_scores(y_actual, method, method2):
    auc_score = roc_auc_score(y_actual, method)
    fpr_df, tpr_df, _ = roc_curve(y_actual, method2)
    return (auc_score, fpr_df, tpr_df)


# Loading the dataset
df = pd.read_csv("Churn_Modelling.csv", delimiter=",")

# Print the shape of the DataFrame
print("DataFrame Shape (rows, columns):")
print(df.shape)
print("\n")  # Adding a new line for better separation

# Check and print columns list and missing values
print("Missing Values by Column:")
print(df.isnull().sum())
print("\n")  # Adding a new line for better separation

# Print unique count for each variable
print("Unique Values Count by Column:")
print(df.nunique())
print("\n")  # Adding a new line for better separation

# Drop the specified columns
df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# Review and print the top rows of what is left of the data frame
print("Data Preview After Dropping Columns:")
print(df.head())


# Creating visualizations to explore the relationship between customer attributes and churn
# These visualizations aim to identify patterns that may inform targeted intervention strategies.

# Pie chart to visualize the overall proportion of churned vs. retained customers.
# This helps in understanding the scale of churn issue at a glance.
labels = "Exited", "Retained"
sizes = [df.Exited[df["Exited"] == 1].count(), df.Exited[df["Exited"] == 0].count()]
explode = (0, 0.1)  # Slightly "explode" the 'Retained' section for better visibility
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.pie(
    sizes, explode=explode, labels=labels, autopct="%1.1f%%", shadow=True, startangle=90
)
ax1.axis("equal")  # Ensure the pie chart is a circle.
plt.title("Proportion of Customer Churned and Retained", size=20)
plt.savefig("customer_churn_proportions.pdf")
plt.close()

# Count plots for categorical variables to explore their influence on customer churn.
# These plots can reveal trends and patterns in how different attributes correlate with churn.
fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
sns.countplot(x="Geography", hue="Exited", data=df, ax=axarr[0][0])
sns.countplot(x="Gender", hue="Exited", data=df, ax=axarr[0][1])
sns.countplot(x="HasCrCard", hue="Exited", data=df, ax=axarr[1][0])
sns.countplot(x="IsActiveMember", hue="Exited", data=df, ax=axarr[1][1])
plt.savefig("customer_churn_relations_categorical.pdf")
plt.close()


# Relations based on the continuous data attributes
fig, axarr = plt.subplots(3, 2, figsize=(20, 12))
sns.boxplot(y="CreditScore", x="Exited", hue="Exited", data=df, ax=axarr[0][0])
sns.boxplot(y="Age", x="Exited", hue="Exited", data=df, ax=axarr[0][1])
sns.boxplot(y="Tenure", x="Exited", hue="Exited", data=df, ax=axarr[1][0])
sns.boxplot(y="Balance", x="Exited", hue="Exited", data=df, ax=axarr[1][1])
sns.boxplot(y="NumOfProducts", x="Exited", hue="Exited", data=df, ax=axarr[2][0])
sns.boxplot(y="EstimatedSalary", x="Exited", hue="Exited", data=df, ax=axarr[2][1])
plt.savefig("customer_churn_relations_continuous.pdf")
plt.close()


df["BalanceSalaryRatio"] = df.Balance / df.EstimatedSalary
sns.boxplot(y="BalanceSalaryRatio", x="Exited", hue="Exited", data=df)
plt.ylim(-1, 5)
plt.savefig("BalanceSalaryRatio.pdf")

# Given that tenure is a 'function' of age, we introduce a variable aiming to standardize tenure over age:
df["TenureByAge"] = df.Tenure / (df.Age)
sns.boxplot(y="TenureByAge", x="Exited", hue="Exited", data=df)
plt.ylim(-1, 1)
plt.savefig("TenureByAge.pdf")

# Introduce a variable to capture credit score given age to take into account credit behaviour visavis adult life
df["CreditScoreGivenAge"] = df.CreditScore / (df.Age)

# Arrange columns by data type for easier manipulation
continuous_vars = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary",
    "BalanceSalaryRatio",
    "TenureByAge",
    "CreditScoreGivenAge",
]
cat_vars = ["HasCrCard", "IsActiveMember", "Geography", "Gender"]
df = df[["Exited"] + continuous_vars + cat_vars]
# For the one hot variables, we change 0 to -1 so that the models can capture a negative relation
# where the attribute in inapplicable instead of 0
df.loc[df.HasCrCard == 0, "HasCrCard"] = -1
df.loc[df.IsActiveMember == 0, "IsActiveMember"] = -1

lst = ["Geography", "Gender"]
remove = list()
for i in lst:
    if df[i].dtype == 'object':
        for j in df[i].unique():
            df[i + "_" + j] = np.where(df[i] == j, 1, -1)
        remove.append(i)
df = df.drop(remove, axis=1)
print(df.head())

minVec = df[continuous_vars].min().copy()
maxVec = df[continuous_vars].max().copy()
df[continuous_vars] = (df[continuous_vars] - minVec) / (maxVec - minVec)

# Calculate the correlation matrix
corr_matrix = df[continuous_vars].corr()

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix of Dataset')
plt.savefig("correlation_matrix.pdf")

# Split Train, test data
df_train = df.sample(frac=0.8, random_state=200)
df_test = df.drop(df_train.index)
print(len(df_train))
print(len(df_test))

if optimize:
    # Fit primal logistic regression
    print("Logistic Regression")
    param_grid = {
        "C": [0.1, 0.5, 1, 10, 50, 100],
        "max_iter": [250],
        "fit_intercept": [True],
        "intercept_scaling": [1],
        "penalty": ["l2"],
        "tol": [0.00001, 0.0001, 0.000001],
    }
    log_primal_Grid = GridSearchCV(
        LogisticRegression(solver="lbfgs"), param_grid, cv=10, scoring='f1', refit=True, verbose=0
    )
    log_primal_Grid.fit(df_train.loc[:, df_train.columns != "Exited"], df_train.Exited)
    best_model(log_primal_Grid)

    # Fit logistic regression with degree 2 polynomial kernel
    print("Logistic Regression 2 polynomial kernel")
    param_grid = {
        "C": [0.1, 10, 50],
        "max_iter": [300, 500],
        "fit_intercept": [True],
        "intercept_scaling": [1],
        "penalty": ["l2"],
        "tol": [0.0001, 0.000001],
    }
    poly2 = PolynomialFeatures(degree=2)
    df_train_pol2 = poly2.fit_transform(df_train.loc[:, df_train.columns != "Exited"])
    log_pol2_Grid = GridSearchCV(
        LogisticRegression(solver="liblinear"), param_grid, cv=5, scoring='f1', refit=True, verbose=0
    )
    log_pol2_Grid.fit(df_train_pol2, df_train.Exited)
    best_model(log_pol2_Grid)

    # Fit SVM with RBF Kernel
    print("SVM with RBF Kernel")
    param_grid = {
        "C": [0.5, 100, 150],
        "gamma": [0.1, 0.01, 0.001],
        "probability": [True],
        "kernel": ["rbf"],
    }
    SVM_grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1', refit=True, verbose=0)
    SVM_grid.fit(df_train.loc[:, df_train.columns != "Exited"], df_train.Exited)
    best_model(SVM_grid)

    # Fit SVM with pol kernel
    print("SVM with pol Kernel")
    param_grid = {
        "C": [0.5, 1, 10, 50, 100],
        "gamma": [0.1, 0.01, 0.001],
        "probability": [True],
        "kernel": ["poly"],
        "degree": [2, 3],
    }
    SVM_grid = GridSearchCV(SVC(), param_grid, cv=5,scoring='f1', refit=True, verbose=0)
    SVM_grid.fit(df_train.loc[:, df_train.columns != "Exited"], df_train.Exited)
    best_model(SVM_grid)

    # Fit random forest classifier
    print("random forest")
    param_grid = {
        "max_depth": [3, 5, 6, 7, 8],
        "max_features": [2, 4, 6, 7, 8, 9],
        "n_estimators": [50, 100],
        "min_samples_split": [3, 5, 6, 7],
    }
    RanFor_grid = GridSearchCV(
        RandomForestClassifier(), param_grid, cv=5,scoring='f1', refit=True, verbose=0
    )
    RanFor_grid.fit(df_train.loc[:, df_train.columns != "Exited"], df_train.Exited)
    best_model(RanFor_grid)

    # Fit Extreme Gradient boosting classifier
    param_grid = {
        "max_depth": [5, 6, 7, 8],
        "gamma": [0.01, 0.001, 0.001],
        "min_child_weight": [1, 5, 10],
        "learning_rate": [0.05, 0.1, 0.2, 0.3],
        "n_estimators": [5, 10, 20, 100],
    }
    xgb_grid = GridSearchCV(XGBClassifier(), param_grid, cv=5, scoring='f1', refit=True, verbose=0)
    xgb_grid.fit(df_train.loc[:, df_train.columns != "Exited"], df_train.Exited)
    best_model(xgb_grid)
else:
    # Fit primal logistic regression
    print("Logistic regression: ")
    log_primal = LogisticRegression(C=100, class_weight='balanced', dual=False, fit_intercept=True,intercept_scaling=1, max_iter=250,n_jobs=None, 
                                    penalty='l2', random_state=None, solver='lbfgs',tol=1e-05, verbose=0, warm_start=False)
    log_primal.fit(df_train.loc[:, df_train.columns != 'Exited'],df_train.Exited)

    print(
        classification_report(
            df_train.Exited,
            log_primal.predict(df_train.loc[:, df_train.columns != "Exited"]),
        )
    )
    print("\n")

    # Fit logistic regression with pol 2 kernel
    print("Logistic regression with pol2 kernel: ")
    poly2 = PolynomialFeatures(degree=2)
    df_train_pol2 = poly2.fit_transform(df_train.loc[:, df_train.columns != 'Exited'])
    log_pol2 = LogisticRegression(
        C=10,
        class_weight="balanced",
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        max_iter=300,
        n_jobs=None,
        penalty="l2",
        random_state=None,
        solver="liblinear",
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )
    log_pol2.fit(df_train_pol2,df_train.Exited)

    print(classification_report(df_train.Exited, log_pol2.predict(df_train_pol2)))
    print("\n")

    # Fit SVM with RBF Kernel
    print("SVM with RBF Kernel: ")
    SVM_RBF = SVC(
        C=100,
        cache_size=200,
        class_weight="balanced",
        coef0=0.0,
        decision_function_shape="ovr",
        degree=3,
        gamma=0.1,
        kernel="rbf",
        max_iter=-1,
        probability=True,
        random_state=None,
        shrinking=True,
        tol=0.001,
        verbose=False,
    )
    SVM_RBF.fit(df_train.loc[:, df_train.columns != "Exited"], df_train.Exited)

    print(
        classification_report(
            df_train.Exited,
            SVM_RBF.predict(df_train.loc[:, df_train.columns != "Exited"]),
        )
    )
    print("\n")

    # Fit SVM with Pol Kernel
    print("SVM with Pol Kernel: ")
    SVM_POL = SVC(
        C=100,
        cache_size=200,
        class_weight='balanced',
        coef0=0.0,
        decision_function_shape="ovr",
        degree=2,
        gamma=0.1,
        kernel="poly",
        max_iter=-1,
        probability=True,
        random_state=None,
        shrinking=True,
        tol=0.001,
        verbose=False,
    )
    SVM_POL.fit(df_train.loc[:, df_train.columns != "Exited"], df_train.Exited)

    print(
    classification_report(
        df_train.Exited,
        SVM_POL.predict(df_train.loc[:, df_train.columns != "Exited"]),
    )
    )
    print("\n")

    # Fit Random Forest classifier
    print("RandomForestClassifier: ")
    RF = RandomForestClassifier(
        bootstrap=True,
        class_weight="balanced",
        criterion="gini",
        max_depth=8,
        max_features=6,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=1,
        min_samples_split=3,
        min_weight_fraction_leaf=0.0,
        n_estimators=50,
        n_jobs=None,
        oob_score=False,
        random_state=None,
        verbose=0,
        warm_start=False,
    )
    RF.fit(df_train.loc[:, df_train.columns != "Exited"], df_train.Exited)
    print(
        classification_report(
            df_train.Exited, RF.predict(df_train.loc[:, df_train.columns != "Exited"])
        )
    )
    print("\n")

    # Create a SHAP explainer
    explainer = shap.Explainer(RF)

    # Calculate SHAP values - using a sample or the entire training set
    X_sample = df_train.loc[:, df_train.columns != 'Exited'].sample(100, random_state=42)
    shap_values = explainer.shap_values(X_sample)

    # Generate the SHAP summary plot and save it
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)  # `show=False` prevents the plot from being shown immediately

    # Save the figure
    plt.savefig('shap_summary_plot.pdf', bbox_inches='tight')  # Saves the plot as a PDF file. Adjust the filename/path as needed.
    plt.close()  # Closes the plot figure to free up memory

    # Fit Extreme Gradient Boost Classifier
    print("Extreme Gradient Boost Classifier: ")
    XGB = XGBClassifier(
        base_score=0.5,
        booster="gbtree",
        colsample_bylevel=1,
        colsample_bytree=1,
        gamma=0.01,
        learning_rate=0.1,
        max_delta_step=0,
        max_depth=7,
        min_child_weight=5,
        n_estimators=20,
        n_jobs=1,
        nthread=None,
        objective="binary:logistic",
        random_state=0,
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        seed=None,
        subsample=1,
    )
    XGB.fit(df_train.loc[:, df_train.columns != "Exited"], df_train.Exited)

    print(
        classification_report(
            df_train.Exited, XGB.predict(df_train.loc[:, df_train.columns != "Exited"])
        )
    )
    print("\n")

    y = df_train.Exited
    X = df_train.loc[:, df_train.columns != "Exited"]
    X_pol2 = df_train_pol2
    auc_log_primal, fpr_log_primal, tpr_log_primal = get_auc_scores(
        y, log_primal.predict(X), log_primal.predict_proba(X)[:, 1]
    )
    auc_log_pol2, fpr_log_pol2, tpr_log_pol2 = get_auc_scores(
        y, log_pol2.predict(X_pol2), log_pol2.predict_proba(X_pol2)[:, 1]
    )
    auc_SVM_RBF, fpr_SVM_RBF, tpr_SVM_RBF = get_auc_scores(
        y, SVM_RBF.predict(X), SVM_RBF.predict_proba(X)[:, 1]
    )
    auc_SVM_POL, fpr_SVM_POL, tpr_SVM_POL = get_auc_scores(
        y, SVM_POL.predict(X), SVM_POL.predict_proba(X)[:, 1]
    )
    auc_RF, fpr_RF, tpr_RF = get_auc_scores(y, RF.predict(X), RF.predict_proba(X)[:, 1])
    auc_XGB, fpr_XGB, tpr_XGB = get_auc_scores(
        y, XGB.predict(X), XGB.predict_proba(X)[:, 1]
    )

    plt.figure(figsize=(12, 6), linewidth=1)
    plt.plot(
        fpr_log_primal,
        tpr_log_primal,
        label="log primal Score: " + str(round(auc_log_primal, 5)),
    )
    plt.plot(
        fpr_log_pol2, tpr_log_pol2, label="log pol2 score: " + str(round(auc_log_pol2, 5))
    )
    plt.plot(fpr_SVM_RBF, tpr_SVM_RBF, label="SVM RBF Score: " + str(round(auc_SVM_RBF, 5)))
    plt.plot(fpr_SVM_POL, tpr_SVM_POL, label="SVM POL Score: " + str(round(auc_SVM_POL, 5)))
    plt.plot(fpr_RF, tpr_RF, label="RF score: " + str(round(auc_RF, 5)))
    plt.plot(fpr_XGB, tpr_XGB, label="XGB score: " + str(round(auc_XGB, 5)))
    plt.plot([0, 1], [0, 1], "k--", label="Random: 0.5")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curve")
    plt.legend(loc="best")
    plt.savefig('roc_results_ratios.pdf')
    # plt.show()

    print(
        classification_report(
            df_test.Exited, RF.predict(df_test.loc[:, df_test.columns != "Exited"])
        )
    )

    auc_RF_test, fpr_RF_test, tpr_RF_test = get_auc_scores(df_test.Exited, RF.predict(df_test.loc[:, df_test.columns != 'Exited']),
                                                RF.predict_proba(df_test.loc[:, df_test.columns != 'Exited'])[:,1])
    plt.figure(figsize = (12,6), linewidth= 1)
    plt.plot(fpr_RF_test, tpr_RF_test, label = 'RF score: ' + str(round(auc_RF_test, 5)))
    plt.plot([0,1], [0,1], 'k--', label = 'Random: 0.5')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.savefig('roc_results_ratios_test.pdf')
