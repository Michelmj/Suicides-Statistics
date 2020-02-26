#!/usr/bin/env python
# coding: utf-8
# IMPORTING LIBRARIES
# In[1]:


get_ipython().system('pip install bubbly')


# In[2]:


# Basic operations
import numpy as np
import pandas as pd

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Interactive visualizations
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot 
init_notebook_mode(connected = True)
import plotly.graph_objs as go
py.init_notebook_mode()
from plotly import tools
import plotly.figure_factory as ff

from bubbly.bubbly import bubbleplot
import plotly.tools as tls
from numpy import array
from matplotlib import cm


# In[3]:


get_ipython().system('pip install squarify')

# IMPORTING DATASET
# In[4]:


SS = pd.read_csv("C:/Users/Delmafia91/Downloads/SuicidalStatistics.csv")
SS = SS.sort_values(['year'], ascending = True)
print(SS.shape)


# In[5]:


# Checking total number of countries for suicidal analysis

print('Number of countries for analysis: ', SS['country'].nunique())


# In[6]:


# Head check
ss = ff.create_table(SS.head())
py.iplot(ss)


# In[7]:


ss = ff.create_table(SS.describe())
py.iplot(ss)


# In[8]:


SS.rename({'sex' : 'gender', 'suicides_no' : 'suicides'}, inplace = True, axis = 1)
SS.columns


# In[9]:


# Always check for null values in dataset

SS.isnull().sum()


# In[10]:


# Filling the missing values

SS['suicides'].fillna(0, inplace = True)

# for population, we are going to use the mean
#SS['population'].mean()

SS['population'].fillna(1664090, inplace = True)

# Checking if there is any null values left
SS['suicides'] = SS['suicides'].astype(int)
SS['population'] = SS['population'].astype(int)

# DATA VISUALIZATION
# In[11]:


import warnings
warnings.filterwarnings('ignore')

figure = bubbleplot(dataset = SS,
            x_column = 'suicides',
            y_column = 'population',
            bubble_column = 'country',
            color_column =  'country', 
            x_title = 'Number of Suicides', 
            y_title = 'Population', 
            title = 'Population and Suicides',
            x_logscale = False, 
            scale_bubble= 1, height = 550)

py.iplot(figure, config = {'scrollzom' : True})

# In this plot, regions such as Africa and Asia are high as compared to U.S. and Europe
# In[12]:


# Visualization of the different countries distribution

plt.style.use('seaborn-dark')
plt.rcParams['figure.figsize'] = (15, 9)

color = plt.cm.winter(np.linspace(0, 10, 20))
x = pd.DataFrame(SS.groupby(['country'])['suicides'].sum().reset_index())
x.sort_values(by = ['suicides'], ascending = False, 
              inplace = True)

sns.barplot(x['country'].head(10), 
        y = x['suicides'].head(10), data = x, palette = 'winter')
plt.title('Top 10 Countries in suicides', fontsize = 40)
plt.xlabel('Name of Country', fontsize = 35)
plt.xticks(rotation = 90)
plt.ylabel('Count', fontsize = 35)
plt.show()


# In[13]:


# This is a much clearer representation of the different countries with the most suicidal events


# In[14]:


# Visualization of the different year distribution 

plt.style.use('seaborn-dark')
plt.rcParams['figure.figsize'] = (20, 10)

x = pd.DataFrame(SS.groupby(['year'])['suicides'].sum().reset_index())
x.sort_values(by = ['suicides'], ascending = False, inplace = True)

sns.barplot(x['year'], y = x['suicides'], data = x, palette = 'cool')
plt.title('Distribution of suicides per year', fontsize = 40)
plt.xlabel('year', fontsize = 35)
plt.ylabel('count', fontsize = 35)
plt.xticks(rotation = 90)
plt.show()


# In[15]:


# Gender spectrum

color = plt.cm.Reds(np.linspace(0, 1, 2))
SS['gender'].value_counts().plot.pie(colors = color, 
                    figsize = (10, 10), startangle = 80)
plt.title('Gender', fontsize = 22)
plt.axis('on')
plt.show()


# In[19]:


plt.style.use('seaborn-dark')
plt.rcParams['figure.figsize'] = (20, 10)

x = pd.DataFrame(SS.groupby(['gender'])['suicides'].sum().reset_index())
x.sort_values(by = ['suicides'], ascending = False, inplace = True)

sns.barplot(x['gender'], y = x['suicides'], data = x, palette = 'afmhot')
plt.title('Distribution of Suicides by Gender', fontsize = 40)
plt.xlabel('year', fontsize = 35)
plt.ylabel('count', fontsize = 35)
plt.xticks(rotation = 90)
plt.show()


# In[21]:


# Since we are in the United States, then we will focus here

SS[SS['country'] == 'United States of America'].sample(30)


# In[22]:


# Since we have categorical values in the dataset, it is best 
# replace them with some numerical values

SS['age'] = SS['age'].replace('5-14 years', 0)
SS['age'] = SS['age'].replace('15-24 years', 1)
SS['age'] = SS['age'].replace('25-34 years', 2)
SS['age'] = SS['age'].replace('35-54 years', 3)
SS['age'] = SS['age'].replace('55-74 years', 4)
SS['age'] = SS['age'].replace('75+ years', 5)

s1 = SS[SS['age'] == 0]['suicides'].sum()
s2 = SS[SS['age'] == 1]['suicides'].sum()
s3 = SS[SS['age'] == 2]['suicides'].sum()
s4 = SS[SS['age'] == 3]['suicides'].sum()
s5 = SS[SS['age'] == 4]['suicides'].sum()
s6 = SS[SS['age'] == 5]['suicides'].sum()


# In[27]:


s = pd.DataFrame([s1, s2, s3, s4 ,s5, s6])
s.index = ['5-14', '15-24', '25-34', '35-54', '55-74', '75+']
s.plot(kind = 'bar', color = 'grey')
plt.title('Suicides in different age groups')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.show()

SUICIDE TRENDS ACCORDING TO YEAR
# In[36]:


S = SS.groupby(['country', 'year'])['suicides'].mean()
S = pd.DataFrame(S)

plt.rcParams['figure.figsize'] = (25, 35)
plt.style.use('dark_background')

plt.subplot(3, 1, 1)
color = plt.cm.hot(np.linspace(0, 1, 40))
S['suicides']['United States of America'].plot.bar(color = color)
plt.title('Suicides Trends in USA per Year', fontsize = 40)

plt.subplot(3, 1, 2)
color = plt.cm.spring(np.linspace(0, 1, 40))
S['suicides']['Russian Federation'].plot.bar(color = color)
plt.title('Suicides Trends in Russian Federation per Year', fontsize =40)

plt.subplot(3, 1, 3)
color = plt.cm.PuBu(np.linspace(0, 1, 40))
S['suicides']['Japan'].plot.bar(color = color)
plt.title('Suicides Trends in Japan per Year', fontsize = 40)

plt.show()


# In[39]:


S1 = SS.groupby(['country', 'age'])['suicides'].mean()
S1 = pd.DataFrame(S1)

plt.rcParams['figure.figsize'] = (25, 35)
plt.style.use('dark_background')

plt.subplot(3, 1, 1)
color = plt.cm.hot(np.linspace(0, 1, 10))
S1['suicides']['United States of America'].plot.bar(color = color)
plt.title('Suicides Trends in USA per Age', fontsize = 40)
plt.xticks(rotation = 0)

plt.subplot(3, 1, 2)
color = plt.cm.spring(np.linspace(0, 1, 10))
S1['suicides']['Russian Federation'].plot.bar(color = color)
plt.title('Suicides Trends in Russian Federation per Age', fontsize =40)
plt.xticks(rotation = 0)

plt.subplot(3, 1, 3)
color = plt.cm.PuBu(np.linspace(0, 1, 10))
S1['suicides']['Japan'].plot.bar(color = color)
plt.title('Suicides Trends in Japan per Age', fontsize = 40)
plt.xticks(rotation = 0)

plt.show()

In Conclusion, I think this is self-explanatory