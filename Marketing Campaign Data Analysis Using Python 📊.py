import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

file_path = Path(__file__).parent / "ProjectDA" / "marketing_campaign.csv"
df = pd.read_csv("marketing_campaign.csv")

df.columns = df.columns.str.strip()

required_columns = [
    "Customer_ID", "Age", "Gender", "Income",
    "Spending_Score", "Campaign", "Response", "Website_Visits"
]

if not all(col in df.columns for col in required_columns):
    print("Error: Check your CSV column names.")
    print("Available columns:", df.columns.tolist())
    raise SystemExit

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nTotal Customers:", len(df))
print("Average Age:", round(np.mean(df["Age"]), 2))
print("Average Income:", round(np.mean(df["Income"]), 2))
print("Average Spending Score:", round(np.mean(df["Spending_Score"]), 2))

print("\nCampaign Performance:")
print(df["Campaign"].value_counts())

print("\nCustomer Responses:")
print(df["Response"].value_counts())

print("\nAverage Income by Gender:")
print(df.groupby("Gender")["Income"].mean())

print("\nAverage Spending Score by Campaign:")
print(df.groupby("Campaign")["Spending_Score"].mean())

print("\nCampaign Response Analysis:")
print(pd.crosstab(df["Campaign"], df["Response"]))

sns.set_theme(style="whitegrid")

sns.countplot(data=df, x="Campaign", hue="Response")
plt.title("Campaign Responses")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

sns.barplot(data=df, x="Campaign", y="Spending_Score")
plt.title("Average Spending Score by Campaign")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

sns.histplot(data=df, x="Income", bins=10, kde=True)
plt.title("Customer Income Distribution")
plt.tight_layout()
plt.show()

sns.scatterplot(
    data=df,
    x="Age",
    y="Spending_Score",
    hue="Response"
)
plt.title("Age vs Spending Score")
plt.tight_layout()
plt.show()

sns.barplot(data=df, x="Gender", y="Income")
plt.title("Average Income by Gender")
plt.tight_layout()
plt.show()

sns.scatterplot(
    data=df,
    x="Website_Visits",
    y="Spending_Score",
    hue="Response"
)
plt.title("Website Visits vs Spending Score")
plt.tight_layout()
plt.show()

print("\nMarketing Campaign Analysis Completed!")
