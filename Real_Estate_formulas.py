import numpy as np 
import pandas as pd
import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import pylab
from scipy import stats
from scipy.stats import probplot

import matplotlib.pyplot as plt


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def ANOVA(X, y, categorical_var):
    group_list = X[categorical_var].value_counts()
    n_min = 30
    a = [i for i in dict(group_list[group_list > n_min]).keys()]
    df = {}
    df1 = {}
    for i in a:
        group_price_list = np.log(y[X[categorical_var] == i].Price).values
        group_mean_price = np.log(y[X[categorical_var] == i].Price).mean()
        group_observations_n = len(y[X[categorical_var] == i])

        df[i] = group_mean_price, group_observations_n
        df1[i] = group_price_list, group_mean_price

    y_all_groups = []
    for item in df1.values():
        y_all_groups = np.concatenate([y_all_groups, item[0]])
    y_mean = y_all_groups.mean()

    SS_b = 0
    n_all_groups = 0
    for group_mean, n in df.values():
        SS_b += n * (group_mean - y_mean) ** 2
        n_all_groups += n

    SS_w = 0
    for group_values, group_mean in df1.values():
        SS_w += ((group_values - group_mean) **2 ).sum()

    k = len(df)
    k1 = k - 1
    k2 = n_all_groups - k

    sigma2_b = SS_b / k1
    sigma2_w = SS_w / k2

    F = sigma2_b / sigma2_w

    alpha = 0.05
    t = stats.f.ppf(1 - alpha, k1, k2)    

    
    print(f'nummber of {categorical_var} = {X[categorical_var].nunique()}')
    print(f'number of cases with no {categorical_var} specified: {len(X[X[categorical_var].isnull()])}')
    print(f'number of {categorical_var} with more than {n_min} observations = {len(df)}')
    print(f"F-test statistic for log(Price) = {round(F, 3)}, 5% quantile = {round(t, 3)}")
    
    if F > t:
        print(f'Variance of mean log(Price) by {categorical_var} is statistically significant')
    else:
        print(f'Variance of mean log(Price) by {categorical_var} is statistically insignificant')
    
    
    df = pd.DataFrame(df).T
    df.columns = ['mean_price', 'n_observations']
    df.mean_price = round(np.exp(df.mean_price), 2)
    df.n_observations = df.n_observations.astype(int)
    df = df.sort_values(by=['mean_price'], ascending=False)
    df.index.name = categorical_var
    
    return df


def norm_test1(X, keys):
    fig, axes = plt.subplots(ncols=len(keys))
    fig.set_size_inches(4 * len(keys), 4)
    axes = axes.flatten()

    for key, ax in zip(keys, axes):
        ax.hist(X[key], density=True)

        loc = X[key].mean()
        scale = X[key].std()

        x_left, x_right = ax.get_xlim()
        x = np.linspace(x_left, x_right, 10000)
        y = stats.norm.pdf(x, loc=loc, scale=scale)

        ax.plot(x, y, linestyle='dashed')
        ax.set_title(key)
        
def norm_test2(X, keys):
    fig, axes = plt.subplots(ncols=len(keys))
    fig.set_size_inches(4 * len(keys), 4)
    axes = axes.flatten()

    for key, ax in zip(keys, axes):
        samples = X[key]

        loc = samples.mean()
        scale = samples.std()

        interval = np.linspace(0, 1, samples.shape[0])[1:-1]
        x = stats.norm.ppf(interval, loc=loc, scale=scale)
        y = np.quantile(samples, interval)

        ax.scatter(x, y, s=5)
        ax.plot(x, x, color='C1', linestyle='dashed')

        ax.set_title(key)
        ax.set_xlabel('theoretical quantiles')
        ax.set_ylabel('sample quantiles')
        
def norm_test3(X, keys):
    for key in keys:
        print(key)

        samples = X[key]

        loc = samples.mean()
        scale = samples.std()

        for i in range(1, 4):
            true_value = stats.norm.cdf(i) - stats.norm.cdf(-i)
            sample_value = ((samples >= loc - i * scale) & (samples <= loc + i * scale)).sum() / samples.shape[0]

            print(f'{i} sigma(s)')
            print(f'\ttheoretical:\t{true_value}')
            print(f'\tsample:\t\t{sample_value}')

        print()    


class FeatureImputer:
    """Заполнение пропусков и облработка выбросов"""
    
    def __init__(self):
        self.medians = None
        self.Square_medians = None        

        
    def fit(self, X):
        self.medians = X.median()
        self.Square_medians = X.groupby(['DistrictId', 'HouseYear','Rooms'])['Square'].median()        

    
    def transform(self, X):
        
        # Rooms        
        X.loc[X['Rooms'] == 0, 'Rooms'] = 1
        X.loc[X['Rooms'] >= 6, 'Rooms'] = self.medians['Rooms']
        
        # Square
        X.loc[((X.Rooms < 4) | (X.KitchenSquare <=10)) & (X.Square > 250), 'Square'] /= 10
        X.loc[(X.Square < 13) & (X.LifeSquare > 13), 'Square'] = X[(X.Square < 13) & 
                                                                   (X.LifeSquare > 13)].LifeSquare
        for i in X[(X.Square < 13) | X.Square.isnull()].index:
            X.loc[i, 'Square'] = self.Square_medians[(X.loc[i, 'DistrictId'], 
                                                      X.loc[i, 'HouseYear'], 
                                                      X.loc[i, 'Rooms'])]
        
        # KitchenSquare
        X.loc[X['KitchenSquare'] < 3, 'KitchenSquare'] = 3        
        X.loc[X['KitchenSquare'] > 1000, 'KitchenSquare'] /= 10 
                
        
        # HouseFloor, Floor
        X['HouseFloor_outlier'] = 0
        X.loc[X['HouseFloor'] == 0, 'HouseFloor_outlier'] = 1
        X.loc[X['Floor'] > X['HouseFloor'], 'HouseFloor_outlier'] = 1
        
        X.loc[X['HouseFloor'] == 0, 'HouseFloor'] = self.medians['HouseFloor']
        X.loc[X['Floor'] > X['HouseFloor'], 'Floor'] = X.loc[X['Floor'] > X['HouseFloor'], 'HouseFloor']
        
                
        # HouseYear
        current_year = now = datetime.datetime.now().year
        
        X['HouseYear_outlier'] = 0
        X.loc[X['HouseYear'] > current_year, 'HouseYear_outlier'] = 1
        
        X.loc[X['HouseYear'] > current_year, 'HouseYear'] = current_year
        
        # Healthcare_1
        if 'Healthcare_1' in X.columns:
            X.drop('Healthcare_1', axis=1, inplace=True)
            
        # LifeSquare
        X['LifeSquare_nan'] = X['LifeSquare'].isna() * 1
        
        condition = (X['LifeSquare'].isna() | (X['LifeSquare'] < 8)) &\
                      (~X['Square'].isna()) & \
                      (~X['KitchenSquare'].isna())
        
        X.loc[condition, 'LifeSquare'] = X.loc[condition, 'Square'] - X.loc[condition, 'KitchenSquare']
        
                
        return X
    
 

class FeatureGenetator():
    """Генерация новых фич"""
    
    def __init__(self):
        self.DistrictId_counts = None
        self.binary_to_numbers = None
        self.med_price_by_district = None
        self.med_price_by_ecology_1 = None
        self.med_price_by_shops_1 = None
        self.med_price_by_social_1 = None
        self.med_price_by_social_2 = None
        self.med_price_by_social_3 = None
        self.med_price_by_helthcare_2 = None
        self.med_price_by_floor_year = None
        
        
    def fit(self, X, y=None):
        
        X = X.copy()
        
        # DistrictID
        district = X['DistrictId'].value_counts()
        district = district[district > 50]        
        self.DistrictId_counts = dict(district)
        
        # Ecology_1
        ecology_1 = X['Ecology_1'].value_counts()
        ecology_1 = ecology_1[ecology_1 > 50]        
        self.Ecology_1_counts = dict(ecology_1)
        
        # Shops_1
        shops_1 = X['Shops_1'].value_counts()
        shops_1 = shops_1[shops_1 > 50]        
        self.Shops_1_counts = dict(shops_1)
        
        # Social_1
        social_1 = X['Social_1'].value_counts()
        social_1 = social_1[social_1 > 40]        
        self.Social_1_counts = dict(social_1)
                
        # Social_2
        social_2 = X['Social_2'].value_counts()
        social_2 = social_2[social_2 > 40]        
        self.Social_2_counts = dict(social_2)
        
        # Social_3
        social_3 = X['Social_3'].value_counts()
        social_3 = social_3[social_3 > 40]        
        self.Social_3_counts = dict(social_3)
        
        # Helthcare_2
        helthcare_2 = X['Helthcare_2'].value_counts()
        helthcare_2 = helthcare_2[helthcare_2 > 40]        
        self.Helthcare_2_counts = dict(helthcare_2)
        
       
      
        # Binary features
        self.binary_to_numbers = {'A': 0, 'B': 1}       
        
        
        # Target encoding
        ## 1. District
        df = X.copy()
        
        if y is not None:
            df['Price'] = y.values            
            df['DistrictId_popular'] = df['DistrictId'].copy()
            df.loc[~df['DistrictId_popular'].isin(district.keys().tolist())] = np.nan
            
            self.med_price_by_district = df.groupby(['DistrictId_popular', 'Rooms'], 
                                                    as_index=False).agg({'Price':'median'}).\
                                            rename(columns={'Price':'MedPriceByDistrict',
                                                           'DistrictId_popular': 'DistrictId'}) 
    
        ## 2. Ecology_1    
        if y is not None:
            df['Price'] = y.values            
            df['Ecology_1_frequent'] = df['Ecology_1'].copy()
            df.loc[~df['Ecology_1_frequent'].isin(ecology_1.keys().tolist())] = np.nan            
            self.med_price_by_ecology_1 = df.groupby(['Ecology_1_frequent', 'Rooms'], 
                                                     as_index=False).agg({'Price':'median'}).\
                                            rename(columns={'Price':'MedPriceByEcology_1',
                                                           'Ecology_1_frequent': 'Ecology_1'})
        ## 2. Shops_1    
        if y is not None:
            df['Price'] = y.values            
            df['Shops_1_frequent'] = df['Shops_1'].copy()
            df.loc[~df['Shops_1_frequent'].isin(shops_1.keys().tolist())] = np.nan            
            self.med_price_by_shops_1 = df.groupby(['Shops_1_frequent', 'Rooms'], 
                                                     as_index=False).agg({'Price':'median'}).\
                                            rename(columns={'Price':'MedPriceByShops_1',
                                                           'Shops_1_frequent': 'Shops_1'})

        ## 3.1 Social_1
        if y is not None:
            df['Price'] = y.values            
            df['Social_1_frequent'] = df['Social_1'].copy()
            df.loc[~df['Social_1_frequent'].isin(social_1.keys().tolist())] = np.nan            
            self.med_price_by_social_1 = df.groupby(['Social_1_frequent', 'Rooms'], 
                                                    as_index=False).agg({'Price':'median'}).\
                                            rename(columns={'Price':'MedPriceBySocial_1',
                                                          'Social_1_frequent': 'Social_1'})
        ## 3.2 Social_2
        if y is not None:
            df['Price'] = y.values            
            df['Social_2_frequent'] = df['Social_2'].copy()
            df.loc[~df['Social_2_frequent'].isin(social_2.keys().tolist())] = np.nan            
            self.med_price_by_social_2 = df.groupby(['Social_2_frequent', 'Rooms'], 
                                                    as_index=False).agg({'Price':'median'}).\
                                            rename(columns={'Price':'MedPriceBySocial_2',
                                                          'Social_2_frequent': 'Social_2'})
        ## 3.3 Social_3
        if y is not None:
            df['Price'] = y.values            
            df['Social_3_frequent'] = df['Social_3'].copy()
            df.loc[~df['Social_3_frequent'].isin(social_3.keys().tolist())] = np.nan            
            self.med_price_by_social_3 = df.groupby(['Social_3_frequent', 'Rooms'], 
                                                    as_index=False).agg({'Price':'median'}).\
                                            rename(columns={'Price':'MedPriceBySocial_3',
                                                          'Social_3_frequent': 'Social_3'})            
        ## 4. Helthcare_2
        if y is not None:
            df['Price'] = y.values            
            df['Helthcare_2_frequent'] = df['Helthcare_2'].copy()
            df.loc[~df['Helthcare_2_frequent'].isin(helthcare_2.keys().tolist())] = np.nan            
            self.med_price_by_helthcare_2 = df.groupby(['Helthcare_2_frequent', 'Rooms'], 
                                                    as_index=False).agg({'Price':'median'}).\
                                            rename(columns={'Price':'MedPriceByHelthcare_2',
                                                          'Helthcare_2_frequent': 'Helthcare_2'})
        ## 5. floor, year
        if y is not None:
            df['Price'] = y.values
            df = self.floor_to_cut(df)
            df = self.year_to_cut(df)
            self.med_price_by_floor_year = df.groupby(['year_cut', 'floor_cut'], 
                                                      as_index=False).agg({'Price':'median'}).\
                                            rename(columns={'Price':'MedPriceByFloorYear'})
        
    def transform(self, X):
        
        # DistrictId
        X['DistrictId_count'] = X['DistrictId'].map(self.DistrictId_counts)
        
        X['new_district'] = 0
        X.loc[X['DistrictId_count'].isna(), 'new_district'] = 1
        
        X['DistrictId_count'].fillna(5, inplace=True)
        
        # Binary features
        X['Ecology_2'] = X['Ecology_2'].map(self.binary_to_numbers)
        X['Ecology_3'] = X['Ecology_3'].map(self.binary_to_numbers)
        X['Shops_2'] = X['Shops_2'].map(self.binary_to_numbers)
        
        # More categorical features
        X = self.floor_to_cut(X)
        X = self.year_to_cut(X)         
        
        # Target encoding
        if self.med_price_by_district is not None:
            X = X.merge(self.med_price_by_district, on=['DistrictId', 'Rooms'], how='left')
            
        if self.med_price_by_ecology_1 is not None:
            X = X.merge(self.med_price_by_ecology_1, on=['Ecology_1', 'Rooms'], how='left')
        
        if self.med_price_by_shops_1 is not None:
            X = X.merge(self.med_price_by_shops_1, on=['Shops_1', 'Rooms'], how='left')
            
        if self.med_price_by_social_1 is not None:
            X = X.merge(self.med_price_by_social_1, on=['Social_1', 'Rooms'], how='left')
        
        if self.med_price_by_social_2 is not None:
            X = X.merge(self.med_price_by_social_2, on=['Social_2', 'Rooms'], how='left')
            
        if self.med_price_by_social_3 is not None:
            X = X.merge(self.med_price_by_social_3, on=['Social_3', 'Rooms'], how='left')
            
        if self.med_price_by_helthcare_2 is not None:
            X = X.merge(self.med_price_by_helthcare_2, on=['Helthcare_2', 'Rooms'], how='left')
            
        if self.med_price_by_floor_year is not None:
            X = X.merge(self.med_price_by_floor_year, on=['year_cut', 'floor_cut'], how='left')            
                
        # PCA variables
        X = self.pca_features(X) 
        
        # Dummy variables
        X = self.get_dummies(X) 
                
        return X
    
    @staticmethod
    def floor_to_cut(X):
        
        X['floor_cut'] = np.nan
        
        X.loc[X['Floor'] < 3, 'floor_cut'] = 1  
        X.loc[(X['Floor'] >= 3) & (X['Floor'] <= 5), 'floor_cut'] = 2
        X.loc[(X['Floor'] > 5) & (X['Floor'] <= 9), 'floor_cut'] = 3
        X.loc[(X['Floor'] > 9) & (X['Floor'] <= 15), 'floor_cut'] = 4
        X.loc[X['Floor'] > 15, 'floor_cut'] = 5
            
        return X
     
    @staticmethod
    def year_to_cut(X):
        
        X['year_cut'] = np.nan
        
        X.loc[X['HouseYear'] < 1941, 'year_cut'] = 1
        X.loc[(X['HouseYear'] >= 1941) & (X['HouseYear'] <= 1945), 'year_cut'] = 2
        X.loc[(X['HouseYear'] > 1945) & (X['HouseYear'] <= 1980), 'year_cut'] = 3
        X.loc[(X['HouseYear'] > 1980) & (X['HouseYear'] <= 2000), 'year_cut'] = 4
        X.loc[(X['HouseYear'] > 2000) & (X['HouseYear'] <= 2010), 'year_cut'] = 5
        X.loc[(X['HouseYear'] > 2010), 'year_cut'] = 6
            
        return X    
    
                   
    @staticmethod    
    def get_dummies(X):
        
        ## clustered 'Square', 'LifeSquare', 'Rooms', 'KitchenSquare'
        scaler = StandardScaler()
        a = X[['Square', 'LifeSquare', 'Rooms', 'KitchenSquare']].fillna(0) 
        
        a = scaler.fit_transform(a)
        kmeans = KMeans(n_clusters=3, random_state=0).fit(a)
        X.loc[:, 'Cluster_composit'] = kmeans.labels_
        
        a = pd.get_dummies(X.Cluster_composit).rename(columns={0: 'Cluster_1', 1: 'Cluster_2', 2: 'Cluster_3'})
        X = pd.concat([X, a], axis=1).drop('Cluster_composit', axis=1)
        
        return X
    
        
    @staticmethod   
    def pca_features(X):
        scaler = StandardScaler()
        a = X[['Square', 'LifeSquare', 'Rooms', 'KitchenSquare', 'HouseYear',
               'MedPriceByDistrict', 'MedPriceByEcology_1', 'MedPriceByShops_1',
               'MedPriceBySocial_1','MedPriceBySocial_2', 'MedPriceBySocial_3',
               'Helthcare_2', 'MedPriceByFloorYear']].fillna(0)
        a = scaler.fit_transform(a)
        pca = PCA(n_components=4)
        pca.fit(a)    
        X.loc[:, 'pca_composit_1'] = np.dot(a, pca.components_.T)[:, 0]
        X.loc[:, 'pca_composit_2'] = np.dot(a, pca.components_.T)[:, 1]
        X.loc[:, 'pca_composit_3'] = np.dot(a, pca.components_.T)[:, 2]
        X.loc[:, 'pca_composit_4'] = np.dot(a, pca.components_.T)[:, 3]
    
        return X