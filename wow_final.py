import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

wowdata = pd.read_csv('C://Users//nazli//OneDrive//Desktop//DATA//Sublime Text//wowah_data.csv')

wowdata.info()
wowdata.head()

# The columns starts with space character so for removing
wowdata.columns=wowdata.columns.str.replace(' ', '')
print(wowdata.columns)

# Exp data analysis for unique values in each columns
for column in wowdata.columns:
    unique_value = wowdata[column].nunique()
    print(f"{column} has {unique_value} unique values.")


#For every char, the max level reached
maxlevel = wowdata.groupby('char')['level'].max()

print(maxlevel)

# Display the plot
plt.figure(figsize=(20,10))

sns.set(style='whitegrid')

sns.histplot(maxlevel, bins=30, kde=True, color='green')

plt.title('Distribution of Maximum Levels per Character')
plt.xlabel('Max Level')
plt.ylabel('Frequency')

plt.show()

# Main function for the number of counts for every column
def wowdf_feature(feature):
    align_feature = wowdata.groupby('char')[feature].max().value_counts()
    sorted_align_feature = pd.DataFrame(align_feature)
    return sorted_align_feature


# Running the main function for the "level" column
maxlevel_count = wowdf_feature('level')
maxlevel_count = maxlevel_count.reset_index(drop=True)
maxlevel_count['index'] = maxlevel_count.index


# Plotting "level": Dividing levels for deeper insight number of char on the certain levels
maxlevel_count['level_group'] = pd.cut(maxlevel_count.index + 1, 
                                        bins=np.linspace(0, 80, 9), 
                                        labels=['1-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '80'])
plt.figure(figsize=(15, 6))

# Using swarmplot to visualize the data
sns.swarmplot(data=maxlevel_count, x='level_group', y='level', hue='level_group', size=12)

plt.show()


# Running the main function for the "race" column
wowdf_feature('race')

# Plotting Race
race_count = wowdf_feature('race')

plt.figure(figsize=(15, 8))

sns.set_palette('deep')
colors = sns.color_palette()

plt.pie(race_count['race'], labels=race_count.index, autopct='%1.2f%%', colors=colors, wedgeprops={'linewidth': 4}, startangle=90)

plt.title('Distribution of Races')

plt.show()

# Running the main function for the "charclass" column
wowdf_feature('charclass')

# Plotting Class
charclass_count = wowdf_feature('charclass')

plt.figure(figsize=(15, 8))

sns.set_palette('colorblind')
colors = sns.color_palette()

plt.pie(charclass_count['charclass'], labels=charclass_count.index, autopct='%1.2f%%', colors=colors, wedgeprops={'linewidth': 4}, startangle=90)

plt.title('Distribution of CharClass')

plt.show()


# Race and Class selections made when chars are created 
# Race and Class combinations by chars
race_CharClass = wowdata.groupby('char')[['race', 'charclass']].max()
race_CharClass['race+class'] = race_CharClass['race'] + ' ' + race_CharClass['charclass']
print(race_CharClass.head())

comb_race_CharClass = race_CharClass['race+class'].value_counts().reset_index()
comb_race_CharClass.columns = ['race+class', 'count']


plt.figure(figsize=(20,10))
sns.set(style="whitegrid", rc={"lines.linewidth": 3})

ax = sns.barplot(x='count', y='race+class', data=comb_race_CharClass, palette='Spectral', edgecolor='pink')

plt.xlabel('Count', fontsize=24)
plt.ylabel('Races and Classes', fontsize=24)
plt.title('Race and Class Combinations', fontsize=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=14)

# For the percentages
totals = [i.get_width() for i in ax.patches]
total = sum(totals)

for i in ax.patches:
    ax.text(i.get_width(), i.get_y() + 0.9,
            str(round((i.get_width()/total)*100, 2))+'%', fontsize=16,
            color='black')

plt.show()


# Function of level of intervals by day
def map_level_to_interval(level):
    if 1 <= level <= 20:
        return '1-20'
    elif 21 <= level <= 40:
        return '21-40'
    elif 41 <= level <= 60:
        return '41-60'
    elif 61 <= level <= 80:
        return '61-80'

# For daily analysis 
# Create a new column for level interval = linterval
wowdata['linterval'] = wowdata['level'].apply(map_level_to_interval)

# Split timestamp into date and time
wowdata['date'] = pd.to_datetime(wowdata['timestamp']).dt.date

# Count daily active users in each level interval
daily_users = wowdata.groupby(['date', 'linterval'])['char'].nunique().reset_index()

plot_data = daily_users.pivot(index='date', columns='linterval', values='char').fillna(0)

plt.figure(figsize=(15, 8))
sns.lineplot(data=plot_data)
plt.title('Daily User Activity over one-year')
plt.xlabel('Date')
plt.ylabel('Count')
plt.show().

# For monthly analysis 
# 'char' == 'active_users'
daily_users.rename(columns={'char': 'active_users'}, inplace=True)

# Extract month from the timestamp
daily_users['date'] = pd.to_datetime(daily_users['date'])
daily_users['month'] = daily_users['date'].dt.to_period('M')

# Define level intervals
lintervals1 = ['1-10', '11-20', '21-30', '31-40']
lintervals2 = ['41-50', '51-60', '61-70', '71-80']

# Filter data for the first and second set of intervals
daily_users_filtered1 = daily_users[daily_users['linterval'].isin(lintervals1)]
daily_users_filtered2 = daily_users[daily_users['linterval'].isin(lintervals2)]

# Create strip plot for the first set of intervals
plt.figure(figsize=(20, 10))
sns.stripplot(x='month', y='active_users', hue='linterval', data=daily_users_filtered1, jitter=True, dodge=True, palette='dark')
plt.title('Monthly User Activity for Level Intervals 1-40')
plt.xlabel('Level Interval')
plt.ylabel('Number of Active Users')
plt.legend(loc='upper right')
plt.show()

# Create strip plot for the second set of intervals
plt.figure(figsize=(20, 10))
sns.stripplot(x='month', y='active_users', hue='linterval', data=daily_users_filtered2, jitter=True, dodge=True, palette='Set1')
plt.title('Monthly User Activity for Level Intervals 41-80')
plt.xlabel('Level Interval')
plt.ylabel('Number of Active Users')
plt.legend(loc='upper right')
plt.show()


# For hourly analysis 
# Extract hour from the timestamp
wowdata['hour'] = pd.to_datetime(wowdata['timestamp']).dt.hour

# Group by hour, level_interval, and char, then count unique characters
hourly_users = wowdata.groupby(['hour', 'linterval'])['char'].nunique().reset_index()
hourly_users.rename(columns={'char': 'active_users'}, inplace=True)

# Pivot hourly_users for plotting
plot_data = hourly_users.pivot(index='hour', columns='linterval', values='active_users').fillna(0)

# Create stacked histogram
plot_data.plot(kind='bar', stacked=True, figsize=(15, 8), colormap='Set2')

plt.title('Hourly User Activity for All Level Intervals')
plt.xlabel('Hour')
plt.ylabel('Number of Active Users')
plt.xticks(rotation=0) 
plt.show()


# FOR CLUSTERING

# To define the number of cluster: The Elbow Method

# Numerical and Categorical variables
num_vars = ['level', 'guild']
cat_vars = ['race', 'charclass', 'zone']


# Standardize the numerical variables
num_vars_std = StandardScaler().fit_transform(wowdata[num_vars])

# One-hot encode the categorical variables
cat_vars_enc = OneHotEncoder().fit_transform(wowdata[cat_vars]).toarray()

# Combine the preprocessed numerical and categorical variables
data_preprocessed = np.concatenate([num_vars_std, cat_vars_enc], axis=1)

# Define the range of clusters 
num_clusters = range(1, 21)

# Compute and plot the sum of squared distances for each number of clusters
ssd = []
for k in num_clusters:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data_preprocessed)
    ssd.append(kmeans.inertia_)

plt.plot(num_clusters, ssd, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('distances')
plt.title('k')
plt.show()

# Select the feautures for clustering
X = wowdata[['level', 'race', 'charclass', 'zone', 'guild']]

# Define the preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), ['level', 'guild']),  
    (OneHotEncoder(), ['race', 'charclass', 'zone'])  
)

# Number of clusters coming from the elbow method
kmeans = KMeans(n_clusters=4, random_state=0)

# Make the pipeline and fit 
pipe = make_pipeline(preprocessor, kmeans) 
pipe.fit(X)
clusters = pipe.predict(X)


 # Examine the clusters

# Summary statistics
for i in range(kmeans.n_clusters):
    print(f"Cluster {i}")
    print(wowdata[wowdata['cluster'] == i].describe())
    print("\n")

for cluster in wowdata2['cluster'].unique():
    print(f"Cluster {cluster}:")
    print(wowdata[wowdata['cluster'] == cluster]['race'].value_counts())
    print(wowdata[wowdata['cluster'] == cluster]['charclass'].value_counts())

