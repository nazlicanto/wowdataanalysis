# World of Warcraft Character Data Analysis README

The project aims to analyze the dataset of World of Warcraft characters. 

The dataset contains data about the character's attributes.
#### It consists of the following columns:
	char: unique character identifier.
	level: current level of the character.
	race: character’s selected race.
	charclass: character’s class.
	zone: current location in which the character resides.
	guild: unique guild identifier.
	timestamp: date and time when the data was collected.
 
## Analysis and Visualizations
Below are the key analysis and visualizations performed on the dataset:

Unique column value count: A count of unique values in each column of the dataset was calculated to understand the diversity of the dataset.

Max Level Achievement: Analyzed where 38000 characters, where they reached the maximum level to understand the level distribution among the dataset.

Feature Distribution: The main  function wowdf_feature(feature) was created to identify the distribution of a selected feature. The function groups the data by character and selects the maximum value of the chosen feature.

Level Interval Distribution: Using the wowdf_feature(feature) function, identified how many characters reached certain level intervals when the 80 levels were divided into 8 parts.

Race and Class Distribution: The count of characters in each race and class was calculated to understand the popular races and classes among the dataset.

Race & Class Combination: A combined analysis of race and class was performed to identify the most popular combinations among the characters. Both counts and percentages were calculated.

Daily Activity: Tracked the daily playing activity among the level intervals through the year and displayed it in a line plot.

Hourly Activity (Strip Plot): Tracked the hourly playing activity among the level intervals throughout the year and displayed it in a strip plot.

Hourly Activity (Stacked Bar Plot): To provide another perspective, the hourly playing activity among the level intervals was also displayed in a stacked bar

## Clustering
### K-Means Clustering
The K-Means algorithm is used to cluster the characters. The required number of clusters is determined using the Elbow Method.

### Cluster Analysis: 
Each cluster is analyzed to understand its characteristics/profile. 
