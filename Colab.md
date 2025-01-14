
# GitHub Setup

## Initial Setup

First, download your class repository from GitHub, and upload the folder named GH-ML4T-Fall24 into your Google Drive. 

You will need a Personal Access Token (PAT) from your GT GitHub account to authenticate. Below are the instructions for creating one.

https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic

## Ran At Beginning of Each Session

```python
# Mount your Google Drive to access the repository
from google.colab import drive
drive.mount('/content/drive')

# Navigate to the repository folder in Google Drive
%cd /content/drive/MyDrive/GH-ML4T-Fall24

# Configure Git (required for each new Colab session)
!git config --global user.name "<user-name>"
!git config --global user.email "<user-name>@gatech.edu"

# Set the remote URL with your Personal Access Token (replace <your-PAT>)
!git remote set-url origin https://<user-name>:<your-PAT>@github.gatech.edu/<user-name>/GH-ML4T-Fall24.git

# Pull the latest changes from the remote repository
!git pull origin main
```
## Ran as Needed

### Update and Push Changes

```python
# Navigate to the repository folder
%cd /content/drive/MyDrive/GH-ML4T-Fall24

# Add all changes to the staging area (or use specific files/folders)
!git add .

# Commit changes with a descriptive message (example with multiline notes)
!git commit -m """
Refactor and update repository structure

- Added new preprocessing scripts
- Updated Assignment1 notebook with analysis
- Improved README documentation
"""

# Push changes to the remote repository
!git push origin main
```


# Data Preparation

### Imports

```python
import pandas as pd
import numpy as np
import seaborn as sns
```

### Read in Data

```python
# Ensure proper delimiter and header designation if required
df = pd.read_csv('/content/drive/MyDrive/GH-ML4T-Fall24/assignments/data/marketing_campaign.csv', delimiter='\t', header=0)
```

### Missing Values

#### Check for Missing Values

```python
# Check missing values
df.isnull().sum()
```

#### Handling Missing Values

##### Dropping
Getting rid of rows or columns that contain a null value

###### Delete Rows
```python
df.dropna()

# HOWEVER, it can lose a lot of data. Compare original to dropped data
df.shape
df.dropna().shape
```

###### Delete Column-wise (features)
```python
df.dropna(axis=1)

# Can be useful if majority of data in that feature is null or if features dropped are irrelevant enough to get rid of
```
##### Imputing Missing Values
Replace the null values with the mean of the value
###### Mean Value Imputation
```python
# Mean imputation works well with normall distributed data - no skew

# Create new feature with null replaced with mean
df.['<feature>_mean'] = df['<feature>'].fillna(df['<feature>'].mean())
```

###### Median Value Imputation
```python
# Better for normally distributed data

# Create new feature with null replaced with median
df.['<feature>_median'] = df['<feature>'].fillna(df['<feature>'].median())
```

###### Mode Imputation
```python
# Used for categorical variables

# Retrieve mode value of specific variable
mode_value = df[df['<feature>'].notna()]['<feature>'].mode()[0]

# Create new feature with null replaced with mode
df['<feature>_mode'] = df['<feature>'].fillna(mode_value)
```


### Imbalanced Dataset

#### Check for Imbalance
For our classification problem, we want to make sure that the data we use uniformly represents each class. Otherwise, model may bias towards heavier represented class(es)

```python
# Check the counts of each class of our classification feature

df['<feature>'].value_counts()
```

#### Handling Imbalanced Dataset

##### Up Sampling
Increase less predominant class datapoints by artificially creating new ones based on the existing datapoints

```python
# Example where target classes are 0 and 1
# In this example, 0 is majority (900 points) and 1 is minority (100 points)

df_majority = df[df['<target_class>']==0]
df_minority = df[df['<target_class>']==1]


# Create a new dataset of the same size as the majority class
from sklearn.utils import resample

df_minority_upsampled = resample(
df_minority,
replace=True, # Sample with replacement
n_sample=len(df_majority),
random_state=42 # Any random state value to fix seed
)


# Combine class datasets
pd.concat([df_majority, df_minority_upsampled])

# Verify even distribution
df['<feature>'].value_counts()
```

##### Down Sampling
Decreases the predominant class samples to match the size of minority class samples

```python
# Example where target classes are 0 and 1
# In this example, 0 is majority (900 points) and 1 is minority (100 points)

df_majority = df[df['<target_class>']==0]
df_minority = df[df['<target_class>']==1]


# Create a new dataset of the same size as the majority class
from sklearn.utils import resample

df_majority_downsampled = resample(
df_majority,
n_sample=len(df_minority),
random_state=42 # Any random state value to fix seed
)


# Combine class datasets
pd.concat([df_minority, df_majority_downsampled])

# Verify even distribution
df['<feature>'].value_counts()
```

##### SMOTE (Synthetic Minority Oversampling Technique)
Addresses imbalances by generating synthetic instances of the minority class through interpolation, instead of using copies of existing samples

```python
from imblearn.over_sampling import SMOTE

# Transform dataset, where X is features and y is target
# Automatically oversamples the minority size to match the majority size
oversample = SMOTE()

X,y = oversample.fit_resample(final_df[['<feature_1>','<feature_2>']],
final_df['<target_class>']
)

# Combine into one dataset
df1 = pd.DataFrame(X, columns=['<feature_1>','<feature_2>'])
df2 = pd.DataFrame(y, columns=['<target_class>'])
oversample_df = pd.concat([df1, df2], axis=1)
```


### Outliers

#### Create Outlier Bounds

```python
# Retrieve statistical measurements
lst_nums=[-21, 22,42,78,48,30,88,34,89,38,17,70,60,43,532]
minimum,Q1,median,Q3,maximum=np.quantile(lst_nums, [0,0.25,0.5,0.75,1.0])

# Calculate IQR
IQR = Q3 - Q1

# Calculate bounds
lower_bound = Q1 - 1.5 * (IQR)
higher_bound = Q3 + 1.5 * (IQR)

```
#### View Outliers
Any dots/marks outside the box plot whiskers are outliers

```python
sns.boxplot()
```


### Data Encoding
Convert categorical features into numerical values
#### Nominal/One-Hot Encoding (OHE)
Represents categorical features in an n x n matrix, where there is only a single 1 in each row and column, each indicating a different class. Can lead to a sparse matrix and thus overfitting when category has many classes

```python
from sklearn.preprocessing import OneHotEncoder

# Example dataframe
df = pd.DataFrame({
'color': ['red', 'blue', 'green', 'green', 'red', 'blue']
})

# Create OneHotEncoder, fit and transform on data
encoder = OneHotEncoder()
encoded = encoder.fit_transform(df[['color']]).toarray()

encoder_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

# Can encode additional new data 
encoder.transform([['blue']]).toarray()
```

#### Label and Ordinal Encoding

##### Label Encoding
Assigns a unique numerical label to each category. Downside is that model can falsely accuse importance based on the numerical values, whether it's greater or less than

```python
from sklearn.preprocessing import LabelEncoder

# Create LabelEncoder, fit and transform on data
lbl_encoder = LabelEncoder()
lbl_encoder = fit_transform(df[['color']])

# Can encode additional new data 
lbl_encoder.transform([['red']])
```

##### Ordinal Encoding
Allows controlled assignment of values depending on rank of categories

```python
from sklearn.preprocessing import OrdinalEncoder

# Example dataframe
df = pd.DataFrame({
'size': ['small', 'medium', 'large', 'medium', 'small', 'large']
})

# Create OrdinalEncoder, fit and transform on data
encoder = OrdinalEncoder(categories=[['small', 'medium', 'large']])
encoder = fit_transform(df[['size']])

# Can encode additional new data 
encoder.transform([['medium']])
```

#### Target Guided Ordinal Encoding
Encodes categorical variables based on their relationship with the target variable. Useful for a variable with a large number of unique categories

Replaces the categorical variable with a numerical variable based on mean or median of the target variable for that category

```python
# Example dataframe
df = pd.DataFrame({
'city': ['New York', 'London', 'Paris', 'Tokyo', 'New York', 'Paris', ],
'price': [200, 150, 300, 250, 180, 320]
})

# Find means of each category
mean_price = df.groupby('city')['price'].mean().to_dict()

# Create encoded city feature
df['city_encoded'] = df['city'].map(mean_price)


```



# Exploratory Data Analysis

## Viewing Data 

[[Z- Sample IPYNB/Wine Equality EDA.ipynb|Source]]

### Viewing Information
```python
# Summary of the dataset
df.info()

# Descriptive summary of the dataset
df.describe()

# Lists all the columns names
df.columns

# Show unique values of a feature
df['<feature>'].unique()

# View missing values in the dataset
df.isnull().sum()

# View duplicate records
df[df.duplicated()]

# Remove duplicates
df.drop_duplicates(inplace=True) # Inplace makes changes permanent

# View correlation
df.corr()
```

### Visualizing Information

#### Heatmap
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),annot=True)
```

![[Pasted image 20250115035329.png|center]]

#### Value Counts Visual
```python
df.quality.value_counts().plot(kind='bar')
plt.xlabel("Wine Quality")
plt.ylabel("Count")
plt.show()

# Conclusion is imbalanced dataset
```

![[Pasted image 20250115035432.png|center]]

#### Visualize Distribution for All Columns
```python
for column in df.columns:
    sns.histplot(df[column],kde=True)
```

![[Pasted image 20250114115919.png|center]]


#### Pairplot
Good for univariate, bivariate, and multivariate analysis

```python
sns.pairplot(df)
```

![[Pasted image 20250114120121.png|center]]

#### Categorical Plot
```python
sns.catplot(x='quality', y='alcohol', data=df, kind="box")
```

![[Pasted image 20250114120224.png|center]]


#### Scatterplot Relationship
```python
sns.scatterplot(x='alcohol',y='pH',hue='quality',data=df)
```

![[Pasted image 20250114120452.png|center]]
