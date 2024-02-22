import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Setting display options for better readability
pd.options.display.max_rows = None
pd.options.display.max_columns = None

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

# Split Train, test data
df_train = df.sample(frac=0.8, random_state=200)
df_test = df.drop(df_train.index)
print(len(df_train))
print(len(df_test))
