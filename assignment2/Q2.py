import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the data
df = pd.read_csv('/mnt/home/chenwe81/ITM/assignment2/online_shoppers_intention.csv')

# Filter out bounce rates of 0
df = df[df['BounceRates'] != 0]

# Create a boxplot comparing the bounce rate by visitor type
plt.figure(figsize=(10, 6))
sns.boxplot(x='VisitorType', y='BounceRates', data=df)
plt.title('Bounce Rates by Visitor Type')
plt.show()

# Perform a t-test to see if there is a significant difference
new_visitors = df[df['VisitorType'] == 'New_Visitor']['BounceRates']
returning_visitors = df[df['VisitorType'] == 'Returning_Visitor']['BounceRates']
t_stat, p_val = stats.ttest_ind(new_visitors, returning_visitors)

print(f'T-statistic: {t_stat}\nP-value: {p_val}')