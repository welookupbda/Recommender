
# coding: utf-8

# ### https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b
# ### https://surprise.readthedocs.io/en/stable/getting_started.html

# In[1]:


#%matplotlib inline
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
#import matplotlib.pyplot as plt
import numpy as np


# In[2]:


#import EDA_Library as eda


# In[3]:


product_df = pd.read_csv( "/Users/svattiku/Google Drive/BDA/ProjectBDA/Retail_Data/product_data.csv", 
                         dtype={'product_code':str})


# In[4]:


product_df.head(5)


# In[5]:


from datetime import datetime, timedelta
def getProductDataTimePeriod(dataset_df, numOfDays):
    dataset_df['transactionDate'] = pd.to_datetime(dataset_df.transactionDate, format = '%Y-%m-%d')
    dataset_df[dataset_df['transactionDate'] > pd.to_datetime('2017-04-01', format = '%Y-%m-%d')]
    return(dataset_df[dataset_df['transactionDate'] > pd.to_datetime('2017-06-30', format = '%Y-%m-%d')- timedelta(days=numOfDays)])


# In[6]:


listOfProductsToBeOmitted = ["ONION LOOSE","POTATO LOOSE","TOMATO LOOSE","CORIANDER", 
                             "CUCUMBER GREEN LOOSE", "LADYFINGER LOOSE","BOTTLE GOURD LONG","CABBAGE",
                             "CAPSICUM GREEN","Carrot English Loose", "GINGER", "LEMON LOOSE",
                             "SUGAR MEDIUM LOOSE","TATA SALT PP 1Kg","CAULIFLOWER", "FB VG CHILLI RED MIRCHI",
                             "BRINJAL BHARTA PURPLE", "METHI","RIDGE GOURD", "FB SIS NAMKEENS",
                             "BB-CB-20X25X168SWG-Suitable for ROI New","BB-CB-20X25X208SWG NEW","BB-CB-27X30X208SWG NEW",
                             "BANANA ROBUSTA RAW,BB-CB-27X30X168SWG-Suitable for ROI New","GARLIC PREMIUM","GREEN PEAS",
                             "BANANA ROBUSTA RAW","BB-CB-27X30X168SWG-Suitable for ROI New","BROCCOLI","CAPSICUM RED LOOSE",
                             "MUSHROOM BUTTON"
                            ]
       
product_filtered_df = product_df[~product_df['product_description'].isin(listOfProductsToBeOmitted)]


# In[7]:


product_6months_df = getProductDataTimePeriod(product_df, 180)


# In[8]:


product_6months_filtered_df = getProductDataTimePeriod(product_filtered_df, 180)


# In[9]:


len(product_df.product_code.unique())


# In[10]:


len(product_df.product_description.unique())


# In[11]:


product_df['store_description'].unique()


# In[12]:


store_6months_df = product_6months_df[product_6months_df['store_description'] == 'MM-INDORE-MALHAR MEGA MALL']
store_6months_df.head(5)


# In[13]:


store_6months_filtered_df = product_6months_filtered_df[product_6months_filtered_df['store_description'] == 'MM-INDORE-MALHAR MEGA MALL']
store_6months_filtered_df.head(5)


# In[14]:


# Remove the list of obvious items
 
listOfProductsToBeOmitted = ["ONION LOOSE","POTATO LOOSE","TOMATO LOOSE","CORIANDER", 
                             "CUCUMBER GREEN LOOSE", "LADYFINGER LOOSE","BOTTLE GOURD LONG","CABBAGE",
                             "CAPSICUM GREEN","Carrot English Loose", "GINGER", "LEMON LOOSE",
                             "SUGAR MEDIUM LOOSE","TATA SALT PP 1Kg","CAULIFLOWER", "FB VG CHILLI RED MIRCHI",
                             "BRINJAL BHARTA PURPLE", "METHI","RIDGE GOURD", "FB SIS NAMKEENS",
                             "BB-CB-20X25X168SWG-Suitable for ROI New","BB-CB-20X25X208SWG NEW","BB-CB-27X30X208SWG NEW",
                             "BANANA ROBUSTA RAW,BB-CB-27X30X168SWG-Suitable for ROI New","GARLIC PREMIUM","GREEN PEAS",
                             "BANANA ROBUSTA RAW","BB-CB-27X30X168SWG-Suitable for ROI New","BROCCOLI","CAPSICUM RED LOOSE",
                             "MUSHROOM BUTTON"
                            ]
product_filtered_6months_df = store_6months_df[~store_6months_df['product_description'].isin(listOfProductsToBeOmitted)]
store_filtered_6months_df = product_filtered_6months_df


# In[15]:


store_cust_product_6months_df = store_6months_df.groupby(['customerID','product_description']).size().sort_values(ascending=False).reset_index(name='Rating')
store_cust_product_6months_df.head(5)


# In[16]:


from surprise import SVD,SVDpp,SlopeOne,NMF,NormalPredictor, KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise import accuracy


# In[17]:


reader = Reader(rating_scale=(0, 9))


# In[18]:


data_6months = Dataset.load_from_df(store_cust_product_6months_df[['customerID', 'product_description', 'Rating']], reader)


# In[19]:


def EvaluateDifferentAlgorithms():
    benchmark = []
    # Iterate over all algorithms
    for algorithm in [SVD(), SVDpp(), SlopeOne(), NMF(), NormalPredictor(), KNNBaseline(), KNNBasic(), KNNWithMeans(), KNNWithZScore(), BaselineOnly(), CoClustering()]:
        # Perform cross validation
        results = cross_validate(algorithm, data_6months, measures=['RMSE'], cv=3, verbose=False)
    
        # Get results & append algorithm name
        tmp = pd.DataFrame.from_dict(results).mean(axis=0)
        tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
        benchmark.append(tmp)
    
        print(pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse'))

# Uncomment if you want to execute this function        
#EvaluateDifferentAlgorithms()       


# ### SVD gave least RMSE value in the above test

# In[20]:


#print('Using SVD')
#algo = SVD()
#cross_validate(algo, data_6months, measures=['RMSE'], cv=3, verbose=False)


# In[21]:


def get_Iu_6months(uid):
    """ return the number of items rated by given user
    args: 
      uid: the id of the user
    returns: 
      the number of items rated by the user
    """
    try:
        return len(trainset_6months.ur[trainset_6months.to_inner_uid(uid)])
    except ValueError: # user was not part of the trainset
        return 0
    
def get_Ui_6months(iid):
    """ return number of users that have rated given item
    args:
      iid: the raw id of the item
    returns:
      the number of users that have rated the item.
    """
    try: 
        return len(trainset_6months.ir[trainset_6months.to_inner_iid(iid)])
    except ValueError:
        return 0


# In[45]:


def topRecommendedProducts6months(customerID, number):
    all_predictions_df = all_predictions_6months_df
    topN_df = all_predictions_df[all_predictions_df['uid']==customerID][['iid','err']].head(number)
    productList = ""
    n=1
    for record in topN_df.values.tolist():
        productList = productList+"\n<tr>\n<td>"+str(n)+"</td>\n" +                                        "<td>"+record[0]+"</td>\n"+                                        "<td>"+str(round(record[1],2))+"</td>\n"+                                  "</tr>"
        n = n+1
        
    content = """<html>
                 <head>
                 <style>
                 table, th, td {
                 border: 1px solid black;
                 }
                 </style></head>
                 <body>
                 <h1>Recommender System</h>
                 <h2>"""+"Top products recommended for the customer with ID "+str(customerID)+" are: """"</h2>
                 <table style="width:40%">
                 <tr>
                 <th>S.No</th>
                 <th>Product</th> 
                 <th>Err</th>
                 </tr>
                 """+productList+"""
                 </table>
                 </body>
                 </html>"""
    #print(content)
    return(content)


# In[46]:


trainset_6months, testset_6months = train_test_split(data_6months, test_size=0.2)


# In[38]:


print("Applying SVD on training set")
algo = SVD()
predictions_6months = algo.fit(trainset_6months).test(testset_6months)
accuracy.rmse(predictions_6months)


# In[39]:


predictions_6months


# In[40]:


df_6months = pd.DataFrame(predictions_6months, columns=['uid', 'iid', 'rui', 'est', 'details'])
df_6months['Iu'] = df_6months.uid.apply(get_Iu_6months)
df_6months['Ui'] = df_6months.iid.apply(get_Ui_6months)
df_6months['err'] = abs(df_6months.est - df_6months.rui)
best_predictions_6months = df_6months.sort_values(by='err')[:10]
worst_predictions_6months = df_6months.sort_values(by='err')[-10:]


# In[41]:


best_predictions_6months


# In[42]:


all_predictions_6months_df = df_6months.sort_values(by='err')


# In[43]:


#topRecommendedProducts6months('MMID_20453330', 10, all_predictions_6months_df)


# In[48]:


print(topRecommendedProducts6months('MMID_20452739', 10))


# In[31]:


#topRecommendedProducts('MMID_20490135', 10) # After removing obvious items


# In[32]:


#topRecommendedProducts('MMID_20490135', 10) # Before removing obvious items


# In[33]:


#store_cust_product_50_df[store_cust_product_50_df['customerID'] == 'MMID_20490135'].query('product_description =="LIRIL SOAP LEMON 125G"')


# In[34]:


#store_cust_product_50_df[store_cust_product_50_df['customerID'] == 'MMID_20490135']

