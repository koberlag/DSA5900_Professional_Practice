```python
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from yellowbrick.model_selection import RFECV
from yellowbrick.datasets import load_credit
```


```python
pd.set_option('display.max_columns', None)
df = pd.read_csv("phishing_website_dataset_1.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>having_IPhaving_IP_Address</th>
      <th>URLURL_Length</th>
      <th>Shortining_Service</th>
      <th>having_At_Symbol</th>
      <th>double_slash_redirecting</th>
      <th>Prefix_Suffix</th>
      <th>having_Sub_Domain</th>
      <th>SSLfinal_State</th>
      <th>Domain_registeration_length</th>
      <th>Favicon</th>
      <th>port</th>
      <th>HTTPS_token</th>
      <th>Request_URL</th>
      <th>URL_of_Anchor</th>
      <th>Links_in_tags</th>
      <th>SFH</th>
      <th>Submitting_to_email</th>
      <th>Abnormal_URL</th>
      <th>Redirect</th>
      <th>on_mouseover</th>
      <th>RightClick</th>
      <th>popUpWidnow</th>
      <th>Iframe</th>
      <th>age_of_domain</th>
      <th>DNSRecord</th>
      <th>web_traffic</th>
      <th>Page_Rank</th>
      <th>Google_Index</th>
      <th>Links_pointing_to_page</th>
      <th>Statistical_report</th>
      <th>Result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (11055, 32)




```python
df['Result'].value_counts()
```




     1    6157
    -1    4898
    Name: Result, dtype: int64




```python
df.drop(['index', 'Result'], axis=1).apply(pd.value_counts).fillna(0).astype(int)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>having_IPhaving_IP_Address</th>
      <th>URLURL_Length</th>
      <th>Shortining_Service</th>
      <th>having_At_Symbol</th>
      <th>double_slash_redirecting</th>
      <th>Prefix_Suffix</th>
      <th>having_Sub_Domain</th>
      <th>SSLfinal_State</th>
      <th>Domain_registeration_length</th>
      <th>Favicon</th>
      <th>port</th>
      <th>HTTPS_token</th>
      <th>Request_URL</th>
      <th>URL_of_Anchor</th>
      <th>Links_in_tags</th>
      <th>SFH</th>
      <th>Submitting_to_email</th>
      <th>Abnormal_URL</th>
      <th>Redirect</th>
      <th>on_mouseover</th>
      <th>RightClick</th>
      <th>popUpWidnow</th>
      <th>Iframe</th>
      <th>age_of_domain</th>
      <th>DNSRecord</th>
      <th>web_traffic</th>
      <th>Page_Rank</th>
      <th>Google_Index</th>
      <th>Links_pointing_to_page</th>
      <th>Statistical_report</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>-1</th>
      <td>3793</td>
      <td>8960</td>
      <td>1444</td>
      <td>1655</td>
      <td>1429</td>
      <td>9590</td>
      <td>3363</td>
      <td>3557</td>
      <td>7389</td>
      <td>2053</td>
      <td>1502</td>
      <td>1796</td>
      <td>4495</td>
      <td>3282</td>
      <td>3956</td>
      <td>8440</td>
      <td>2014</td>
      <td>1629</td>
      <td>0</td>
      <td>1315</td>
      <td>476</td>
      <td>2137</td>
      <td>1012</td>
      <td>5189</td>
      <td>3443</td>
      <td>2655</td>
      <td>8201</td>
      <td>1539</td>
      <td>548</td>
      <td>1550</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>135</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3622</td>
      <td>1167</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5337</td>
      <td>4449</td>
      <td>761</td>
      <td>0</td>
      <td>0</td>
      <td>9776</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2569</td>
      <td>0</td>
      <td>0</td>
      <td>6156</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7262</td>
      <td>1960</td>
      <td>9611</td>
      <td>9400</td>
      <td>9626</td>
      <td>1465</td>
      <td>4070</td>
      <td>6331</td>
      <td>3666</td>
      <td>9002</td>
      <td>9553</td>
      <td>9259</td>
      <td>6560</td>
      <td>2436</td>
      <td>2650</td>
      <td>1854</td>
      <td>9041</td>
      <td>9426</td>
      <td>1279</td>
      <td>9740</td>
      <td>10579</td>
      <td>8918</td>
      <td>10043</td>
      <td>5866</td>
      <td>7612</td>
      <td>5831</td>
      <td>2854</td>
      <td>9516</td>
      <td>4351</td>
      <td>9505</td>
    </tr>
  </tbody>
</table>
</div>



# Using only features specific to email analysis:


```python
# Load classification dataset
# Drop index, Result, and features that are not directly important to email analysis
X = df.drop(columns=['index','Result', 'SSLfinal_State', 'Domain_registeration_length', 'Favicon', 'port', 'Request_URL', 'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain', 'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page', 'Statistical_report'])
y = df[['Result']]

X.shape
```




    (11055, 9)



### Recursive Feature Elimination

We will use recursive feature elimination (RFE) to find the important features.

Recursive feature elimination is a feature selection method that fits a model and removes the weakest feature (or features) until the specified number of features is reached. Features are ranked by the model’s coef_ or feature_importances_ attributes, and by recursively eliminating a small number of features per loop, RFE attempts to eliminate dependencies and collinearity that may exist in the model.

RFE requires a specified number of features to keep, however it is often not known in advance how many features are valid. To find the optimal number of features cross-validation is used with RFE to score different feature subsets and select the best scoring collection of features. The RFECV visualizer plots the number of features in the model along with their cross-validated test score and variability and visualizes the selected number of features.


```python
cv = StratifiedKFold(10, random_state=42)
visualizer = RFECV(RandomForestClassifier(random_state=42), cv=cv, scoring='f1_weighted')

visualizer.fit(X, y)        # Fit the data to the visualizer
visualizer.show()           # Finalize and render the figure
```


![png](output_9_0.png)





    <matplotlib.axes._subplots.AxesSubplot at 0x7f9be03e8a30>




```python
# Get the coefficients for feature importance, for each feature
feat_importances = visualizer.rfe_estimator_.estimator_.feature_importances_
feat_columns = X.columns[visualizer.support_]

# Use all features for the bar plot
y_pos = np.arange(len(feat_importances))

# Create bars
plt.bar(y_pos, feat_importances)

# Add title and axis names
plt.title('Importance of Email Features')
plt.xlabel('Features')
plt.ylabel('Feature Importance')


plt.savefig('email_feat_only.png', dpi=300, bbox_inches='tight')

# We can see that there are 2 features that are prominent 
# and appear to be the most important.
```


![png](output_10_0.png)



```python
# Display the top features and their importance
pd.DataFrame(feat_importances,
             index = feat_columns,
             columns=['importance']).sort_values('importance', ascending=False)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>having_Sub_Domain</th>
      <td>0.430904</td>
    </tr>
    <tr>
      <th>Prefix_Suffix</th>
      <td>0.327949</td>
    </tr>
    <tr>
      <th>URLURL_Length</th>
      <td>0.051535</td>
    </tr>
    <tr>
      <th>having_IPhaving_IP_Address</th>
      <td>0.051028</td>
    </tr>
    <tr>
      <th>having_At_Symbol</th>
      <td>0.034060</td>
    </tr>
    <tr>
      <th>Shortining_Service</th>
      <td>0.029805</td>
    </tr>
    <tr>
      <th>double_slash_redirecting</th>
      <td>0.028943</td>
    </tr>
    <tr>
      <th>Abnormal_URL</th>
      <td>0.028078</td>
    </tr>
    <tr>
      <th>HTTPS_token</th>
      <td>0.017696</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Top-n features to select
top_n_feat = 3
top_n_feat_idx = sorted(range(len(feat_importances)), key=lambda i: feat_importances[i], reverse=True)[:top_n_feat]
```


```python
# Use only the top_n_features for the bar plot
bars = feat_columns[top_n_feat_idx]
y_pos = np.arange(len(feat_importances[top_n_feat_idx]))

# Create bars
plt.bar(y_pos, feat_importances[top_n_feat_idx])

# Create names on the x-axis
plt.xticks(y_pos, bars)

# Add title and axis names
plt.title('Importance of Top 3 Email Features')
plt.xlabel('Features')
plt.ylabel('Feature Importance')

plt.savefig('top_n_email_feat.png', dpi=300, bbox_inches='tight')
```


![png](output_13_0.png)



```python
# Display the top n features and their importance
pd.DataFrame(feat_importances[top_n_feat_idx],
             index = feat_columns[top_n_feat_idx],
             columns=['importance']).sort_values('importance', ascending=False)

# We can see that 'having_Sub_Domain', 'Prefix_Suffix', and 'URLURL_Length' appear to be the more important features
# of the dataset that contains only columns that could specifically
# be used directly for email analysis.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>having_Sub_Domain</th>
      <td>0.430904</td>
    </tr>
    <tr>
      <th>Prefix_Suffix</th>
      <td>0.327949</td>
    </tr>
    <tr>
      <th>URLURL_Length</th>
      <td>0.051535</td>
    </tr>
  </tbody>
</table>
</div>



##  Sub Domain and Multi Sub Domains (having_Sub_Domain):

Let us assume we have the following link: http://www.hud.ac.uk/students/. A domain name might include the country-code top-level domains (ccTLD), which in our example is “uk”. The “ac” part is shorthand for “academic”, the combined “ac.uk” is called a second-level domain (SLD) and “hud” is the actual name of the domain. To produce a rule for extracting this feature, we firstly have to omit the (www.) from the URL which is in fact a sub domain in itself. Then, we have to remove the (ccTLD) if it exists. Finally, we count the remaining dots. If the number of dots is greater than one, then the URLis classified as “Suspicious” since it has one sub domain. However, if the dots are greater than two, it is classified as “Phishing” since it will have multiple sub domains. Otherwise, if the URL has no subdomains, we will assign “Legitimate” to the feature.

## Adding Prefix or Suffix Separated by (-) to the Domain (Prefix_Suffix):

The dash symbol is rarely used in legitimate URLs. Phishers tend to add prefixes or suffixes separatedby (-) to the domain name so that users feel that they are dealing with a legitimate webpage. For example http://www.Confirme-paypal.com/

## Long URL to Hide the Suspicious Part (URLURL_Length):

Phishers can use long URL to hide the doubtful part in the address bar. For example: http://federmacedoadv.com.br/3f/aze/ab51e2e319e51502f416dbe46b773a5e/?cmd=_home&amp;dispatch=11004d58f5b74f8dc1e7c2e8dd4105e811004d58f5b74f8dc1e7c2e8dd4105e8@phishing.website.html. To ensure accuracy of our study, we calculated the length of URLs in the dataset and produced anaverage URL length. The results showed that if the length of the URL is greater than or equal 54 characters then the URL classified as phishing. By reviewing our dataset we were able to find 1220 URLs lengths equals to 54 or more which constitute 48.8% of the total dataset size.


```python

```

# Using All Features: 


```python
# Load classification dataset
# Drop index, Result, and features that are not directly important to email analysis
X_all = df.drop(columns=['index','Result'])
y_all = df[['Result']]

X_all.shape
```




    (11055, 30)




```python
cv_all = StratifiedKFold(10, random_state=42)
visualizer_all = RFECV(RandomForestClassifier(random_state=42), cv=cv_all, scoring='f1_weighted')

visualizer_all.fit(X_all, y_all)        # Fit the data to the visualizer
visualizer_all.show()           # Finalize and render the figure
```


![png](output_24_0.png)





    <matplotlib.axes._subplots.AxesSubplot at 0x7f9be5dac220>




```python
# Get the coefficients for feature importance, for each feature
feat_importances_all = visualizer_all.rfe_estimator_.estimator_.feature_importances_
feat_columns_all = X_all.columns[visualizer_all.support_]

# Use all important features for the bar plot
y_pos_all = np.arange(len(feat_importances_all))

# Create bars
plt.bar(y_pos_all, feat_importances_all)

# Add title and axis names
plt.title('Importance of All Features')
plt.xlabel('Features')
plt.ylabel('Feature Importance')

plt.savefig('all_feat.png', dpi=300, bbox_inches='tight')

# We can see that there are 2 features that are very prominent and a few that are slightly prominent
# and appear to make up the more important features
```


![png](output_25_0.png)



```python
# Display the top features and their importance
pd.DataFrame(feat_importances_all,
             index = feat_columns_all,
             columns=['importance']).sort_values('importance', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SSLfinal_State</th>
      <td>0.318529</td>
    </tr>
    <tr>
      <th>URL_of_Anchor</th>
      <td>0.262463</td>
    </tr>
    <tr>
      <th>web_traffic</th>
      <td>0.070082</td>
    </tr>
    <tr>
      <th>having_Sub_Domain</th>
      <td>0.060848</td>
    </tr>
    <tr>
      <th>Links_in_tags</th>
      <td>0.041492</td>
    </tr>
    <tr>
      <th>Prefix_Suffix</th>
      <td>0.038782</td>
    </tr>
    <tr>
      <th>SFH</th>
      <td>0.020772</td>
    </tr>
    <tr>
      <th>Request_URL</th>
      <td>0.019452</td>
    </tr>
    <tr>
      <th>Links_pointing_to_page</th>
      <td>0.019059</td>
    </tr>
    <tr>
      <th>Domain_registeration_length</th>
      <td>0.016344</td>
    </tr>
    <tr>
      <th>age_of_domain</th>
      <td>0.015326</td>
    </tr>
    <tr>
      <th>Google_Index</th>
      <td>0.013117</td>
    </tr>
    <tr>
      <th>having_IPhaving_IP_Address</th>
      <td>0.013011</td>
    </tr>
    <tr>
      <th>DNSRecord</th>
      <td>0.012302</td>
    </tr>
    <tr>
      <th>Page_Rank</th>
      <td>0.011816</td>
    </tr>
    <tr>
      <th>URLURL_Length</th>
      <td>0.007807</td>
    </tr>
    <tr>
      <th>HTTPS_token</th>
      <td>0.005995</td>
    </tr>
    <tr>
      <th>Redirect</th>
      <td>0.005527</td>
    </tr>
    <tr>
      <th>having_At_Symbol</th>
      <td>0.005296</td>
    </tr>
    <tr>
      <th>Submitting_to_email</th>
      <td>0.005257</td>
    </tr>
    <tr>
      <th>Shortining_Service</th>
      <td>0.005226</td>
    </tr>
    <tr>
      <th>popUpWidnow</th>
      <td>0.004675</td>
    </tr>
    <tr>
      <th>Statistical_report</th>
      <td>0.004625</td>
    </tr>
    <tr>
      <th>Favicon</th>
      <td>0.004511</td>
    </tr>
    <tr>
      <th>Abnormal_URL</th>
      <td>0.004131</td>
    </tr>
    <tr>
      <th>double_slash_redirecting</th>
      <td>0.003569</td>
    </tr>
    <tr>
      <th>on_mouseover</th>
      <td>0.003398</td>
    </tr>
    <tr>
      <th>Iframe</th>
      <td>0.002624</td>
    </tr>
    <tr>
      <th>port</th>
      <td>0.002259</td>
    </tr>
    <tr>
      <th>RightClick</th>
      <td>0.001704</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Top-n features to select
top_n_feat_all = 3

top_n_feat_all_idx = sorted(range(len(feat_importances_all)), key=lambda i: feat_importances_all[i], reverse=True)[:top_n_feat_all]

```


```python
# Use only the top_n_features for the bar plot
bars_all = feat_columns_all[top_n_feat_all_idx]
y_pos_all = np.arange(len(feat_importances_all[top_n_feat_all_idx]))

# Create bars
plt.bar(y_pos_all, feat_importances_all[top_n_feat_all_idx])

# Create names on the x-axis
plt.xticks(y_pos_all, bars_all)

# Add title and axis names
plt.title('Importance of Top 3 Features')
plt.xlabel('Features')
plt.ylabel('Feature Importance')

plt.savefig('top_n_feat.png', dpi=300, bbox_inches='tight')
```


![png](output_28_0.png)



```python
# Display the top n features and their importance
pd.DataFrame(feat_importances_all[top_n_feat_all_idx],
             index = feat_columns_all[top_n_feat_all_idx],
             columns=['importance']).sort_values('importance', ascending=False)

# We can see that 'SSLfinal_State', 'URL_of_Anchor', and 'web_traffic' appear to be the more important features
# of the full dataset.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SSLfinal_State</th>
      <td>0.318529</td>
    </tr>
    <tr>
      <th>URL_of_Anchor</th>
      <td>0.262463</td>
    </tr>
    <tr>
      <th>web_traffic</th>
      <td>0.070082</td>
    </tr>
  </tbody>
</table>
</div>



## HTTPS (Hyper Text Transfer Protocol with Secure Sockets Layer) (SSLfinal_State):

The existence of HTTPS is very important in giving the impression of website legitimacy, but this is clearly not enough. The authors in [ CITATION Ram12 \l 1033 ][ CITATION Moh \l 1033 ] suggest checking the certificate assigned with HTTPS including the extent of the trust certificate issuer, and the certificate age. Certificate Authorities that are consistently listed among the top trustworthy names include: “GeoTrust, GoDaddy, Network Solutions, Thawte, Comodo, Doster and VeriSign”. Furthermore, by testing out our datasets, we find that the minimum age of a reputable certificate is two years.

## URL of Anchor:

An anchor is an element defined by the `<a>` tag. This feature examines:
    1. If the <a> tags and the website have different domain
       names. 
    2. If the anchor does not link to any webpage, e.g.:
        A. <a href=“#”>
        B. <a href=“#content”>
        C. <a href=“#skip”>
        D. <a href=“JavaScript ::void(0)”>

## Website Traffic:

This feature measures the popularity of the website by determining the number of visitors and the number of pages they visit. However, since phishing websites live for a short period of time, they may
not be recognized by the Alexa database. By reviewing our dataset, we find that in worst case scenarios, legitimate websites ranked among the top 100,000. Furthermore, if the domain has no traffic or is not recognized by the Alexa database, it is classified as “Phishing”. Otherwise, it is classified as “Suspicious”.


```python

```


```python

```
