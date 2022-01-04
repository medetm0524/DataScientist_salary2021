# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt


df = pd.read_csv('data_cleaned_2021.csv')


#%%
df['Avg Salary(K)'].plot(kind='hist')
df_describe = df.describe()

#%%
plt.figure(figsize=[20,20])
bins = np.arange(df['Avg Salary(K)'].min(), df['Avg Salary(K)'].max(), 5) 
sb.displot(data = df, x = 'Avg Salary(K)', bins = bins)
plt.title('Data Scienties avg salary')

#%%
plt.figure(figsize=[15,8])
df.Sector.value_counts().plot(kind='bar')
plt.xticks(rotation = 90)
plt.title('Data Scientist job posting group by sectors', fontsize = 18)
plt.xlabel('Sector', fontsize = 12)
plt.ylabel('Count', fontsize = 12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12);

#%%
plt.figure(figsize=[15,8])
df['Job Location'].value_counts().plot(kind='bar')
plt.xticks(rotation = 90)
plt.title('Data Science count based on location',fontsize = 18)
plt.xlabel('Location',fontsize = 12)
plt.ylabel('Count',fontsize = 12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12);

#%%
plt.figure(figsize = [30, 15])

plt.subplot(1, 2, 1)
order = ['1 - 50 ', '51 - 200 ', '201 - 500 ', '501 - 1000 ','1001 - 5000 ', '5001 - 10000 ', '10000+ ', 'unknown']
#df['Size'].value_counts().plot(kind='bar')

base_color = sb.color_palette()[0]
sb.countplot(data = df, x = 'Size', color = base_color, order = order)
plt.xticks(rotation = 90)
plt.title("Data Scientist job posting group by Companys' size",fontsize=25)
plt.xlabel("Company's Size",fontsize=18)
plt.ylabel('Count',fontsize=18)
plt.xticks(rotation = 75, fontsize=18)
plt.yticks(fontsize=18);

plt.subplot(1, 2, 2)
order = ['Less than $1 million (USD)','$1 to $5 million (USD)','$5 to $10 million (USD)', '$10 to $25 million (USD)', '$25 to $50 million (USD)', '$50 to $100 million (USD)',
         '$100 to $500 million (USD)', '$500 million to $1 billion (USD)', '$1 to $2 billion (USD)', '$2 to $5 billion (USD)',
         '$10+ billion (USD)', 'Unknown / Non-Applicable']

base_color = sb.color_palette()[0]
sb.countplot(data = df, x = 'Revenue', color = base_color, order = order)
plt.xticks(rotation = 90)
plt.title("Data Science count based on company's revenue",fontsize=25)
plt.xlabel("Company's revenue",fontsize=18)
plt.ylabel('Count',fontsize=18)
plt.xticks(rotation = 75, fontsize=18)
plt.yticks(fontsize=18);

#%%
plt.figure(figsize=[15,8])
order = ['1 - 50 ', '51 - 200 ', '201 - 500 ', '501 - 1000 ','1001 - 5000 ', '5001 - 10000 ', '10000+ ', 'unknown']
#df['Size'].value_counts().plot(kind='bar')

base_color = sb.color_palette()[0]
sb.countplot(data = df, x = 'Size', color = base_color, order = order)
plt.xticks(rotation = 90)
plt.title('Data Science count based company size')
plt.xlabel('Company size')
plt.ylabel('Count');

#%%
plt.figure(figsize=[15,8])
order = ['Less than $1 million (USD)','$1 to $5 million (USD)','$5 to $10 million (USD)', '$10 to $25 million (USD)', '$25 to $50 million (USD)', '$50 to $100 million (USD)',
         '$100 to $500 million (USD)', '$500 million to $1 billion (USD)', '$1 to $2 billion (USD)', '$2 to $5 billion (USD)',
         '$10+ billion (USD)', 'Unknown / Non-Applicable']

base_color = sb.color_palette()[0]
sb.countplot(data = df, x = 'Revenue', color = base_color, order = order)
plt.xticks(rotation = 90)
plt.title("Data Science count based on company's revenue")
plt.xlabel("Company's revenue")
plt.ylabel('Count');

#%%plt.figure(figsize=[15,8])
sb.boxplot(data = df, x = 'Job Location', y = 'Avg Salary(K)', color = base_color);

#%%
g = sb.FacetGrid(data = df, col = 'Job Location', col_wrap = 5);
g.map(plt.hist, 'Avg Salary(K)');

#%%
g = sb.FacetGrid(data = df, col = 'Sector', col_wrap = 5);
g.map(plt.hist, 'Avg Salary(K)');

#%%
sb.barplot(data = df, x = 'Job Location', y = 'Avg Salary(K)', color = base_color)
plt.xticks(rotation = 90);

#%%
plt.figure(figsize=[15,8])
df.groupby('Job Location')['Avg Salary(K)'].mean().sort_values(ascending=False).plot(kind = 'bar')
plt.title('Data Scientist avg salary on various states',fontsize=18)
plt.xlabel('State',fontsize=12)
plt.ylabel('Avg Salary',fontsize=12);
plt.xticks(rotation = 75, fontsize=12)
plt.yticks(fontsize=12)
#df.sort_values('Avg Salary(K)',ascending = False)
#%%
plt.figure(figsize=[15,8])
df.groupby('Sector')['Avg Salary(K)'].mean().sort_values(ascending=False).plot(kind = 'bar')
plt.title('Data Scientist avg salary on different sector',fontsize=18)
plt.xlabel('Sector',fontsize=12)
plt.ylabel('Avg Salary (K)',fontsize=12);
plt.xticks(rotation = 75, fontsize=14)
plt.yticks(fontsize=12)
#%%
plt.figure(figsize=[15,8])
order = ['1 - 50 ', '51 - 200 ', '201 - 500 ', '501 - 1000 ','1001 - 5000 ', '5001 - 10000 ', '10000+ ', 'unknown']

sb.boxplot(x="Size", y="Avg Salary(K)", data=df, whis=np.inf, order = order)
sb.stripplot(x="Size", y="Avg Salary(K)", data=df, color=".3", order = order)
plt.xticks(rotation = 60)
plt.title('Average Salary grouped by company size',fontsize=18)
plt.xlabel('Company size',fontsize=12)
plt.ylabel('Average Salary (K)',fontsize=12)
plt.xticks(rotation = 75, fontsize=12)
plt.yticks(fontsize=12)

#%%
plt.figure(figsize=[15,8])
order = ['Less than $1 million (USD)','$1 to $5 million (USD)','$5 to $10 million (USD)', '$10 to $25 million (USD)', '$25 to $50 million (USD)', '$50 to $100 million (USD)',
         '$100 to $500 million (USD)', '$500 million to $1 billion (USD)', '$1 to $2 billion (USD)', '$2 to $5 billion (USD)',
         '$10+ billion (USD)', 'Unknown / Non-Applicable']

base_color = sb.color_palette()[0]

sb.boxplot(x="Revenue", y="Avg Salary(K)", data=df, whis=np.inf, order = order)
sb.stripplot(x="Revenue", y="Avg Salary(K)", data=df, color=".3", order = order)
plt.xticks(rotation = 60)
plt.title('Average Salary grouped by company revenue',fontsize=18)
plt.xlabel('Company revenue',fontsize=12)
plt.ylabel('Average Salary (K)',fontsize=12)
plt.xticks(rotation = 75, fontsize=12)
plt.yticks(fontsize=12)


#%%
state_order = df.groupby('Job Location')['Avg Salary(K)'].mean().sort_values(ascending=False).index.tolist()

plt.figure(figsize=[15,8])
base_color = sb.color_palette()[0]

sb.boxplot(x="Job Location", y="Avg Salary(K)", data=df, order = state_order)
sb.stripplot(x='Job Location', y="Avg Salary(K)", data=df, color=".3", order = state_order)
plt.xticks(rotation = 60)
plt.title('Average Salary grouped by states',fontsize=18)
plt.xlabel('States',fontsize=12)
plt.ylabel('Average Salary (K)',fontsize=12)
plt.xticks(rotation = 75, fontsize=12)
plt.yticks(fontsize=12)

#%%
plt.figure(figsize = [15,8])
g = sb.FacetGrid(data = df, hue = 'Revenue',height = 4, aspect = 1.5,
                hue_order = ['Less than $1 million (USD)','$1 to $5 million (USD)','$5 to $10 million (USD)', '$10 to $25 million (USD)', '$25 to $50 million (USD)', '$50 to $100 million (USD)',
                             '$100 to $500 million (USD)', '$500 million to $1 billion (USD)', '$1 to $2 billion (USD)', '$2 to $5 billion (USD)',
                             '$10+ billion (USD)', 'Unknown / Non-Applicable'],palette = 'viridis_r')
g.map(sb.regplot, 'Rating', 'Avg Salary(K)', x_jitter = 0.04, fit_reg = False)
plt.xlabel('Rating')
plt.ylabel('Avg Salary (K)');
g.add_legend();

#%%
cm_skills = ['Python','spark', 'aws', 'excel', 'sql', 'sas', 'keras', 'pytorch', 'scikit',
                'tensor', 'hadoop', 'tableau', 'bi', 'flink', 'mongo', 'google_an']
skills_df = df.groupby('job_title_sim')[cm_skills].sum()
skills_df.style.background_gradient(axis='columns')


