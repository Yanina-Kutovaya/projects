#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore')


class FeatureGenetator():
    """Generation of new features"""
    
    def __init__(self):
        self.countries_counts = None
        self.designations_counts = None
        self.provinces_counts = None
        self.region_1_counts = None
        self.region_2_counts = None
        self.varieties_counts = None
        self.wineries_counts = None        
        
        self.med_price_by_country = None
        self.med_price_by_designation = None
        self.med_price_by_province = None
        self.med_price_by_region = None        
        self.med_price_by_variety = None
        self.med_price_by_winery = None       
               
        
    def fit(self, X, y=None):        
        X = X.copy()
        min_n = 25
        
        # Countries
        countries = X.country.value_counts()
        countries = countries[countries > min_n]                
        self.countries_counts = dict(countries)
        
        # Designations
        designations = X.designation.value_counts()
        designations = designations[designations > min_n]                
        self.designations_counts = dict(designations)        
                
        # Provinces
        provinces = X.province.value_counts()
        provinces = provinces[provinces > min_n]                
        self.provinces_counts = dict(provinces)
        
        # Region_1
        region_1 = X.region_1.value_counts()
        region_1 = region_1[region_1 > min_n]                
        self.region_1_counts = dict(region_1)
        
        # Region_2
        region_2 = X.region_2.value_counts()
        region_2 = region_2[region_2 > min_n]                
        self.region_2_counts = dict(region_2)
        
        # Varieties
        varieties = X.variety.value_counts()
        varieties = varieties[varieties > min_n]                
        self.varieties_counts = dict(varieties)
        
        # Wineries
        wineries = X.winery.value_counts()
        wineries = wineries[wineries > min_n]                
        self.wineries_counts = dict(wineries)
        
        # Target encoding
        ## 1. Countries
        df = X.copy()
        
        if y is not None:
            df['price'] = y.values            
            df['countries_popular'] = df['country'].copy()
            df.loc[~df['countries_popular'].isin(countries.keys().tolist())] = np.nan
            
            self.med_price_by_country = df.groupby(['countries_popular'], 
                                                    as_index=False).agg({'price':'median'}).\
                                            rename(columns={'price':'MedPriceByCountry',
                                                           'countries_popular': 'country'})            
        ## 2. Designations    
        if y is not None:
            df['price'] = y.values            
            df['designations_frequent'] = df['designation'].copy()
            df.loc[~df['designations_frequent'].isin(designations.keys().tolist())] = np.nan
            
            self.med_price_by_designation = df.groupby(['designations_frequent'], 
                                                     as_index=False).agg({'price':'median'}).\
                                            rename(columns={'price':'MedPriceByDesignation',
                                                           'designations_frequent': 'designation'})            
        ## 3. Provinces    
        if y is not None:
            df['price'] = y.values            
            df['provinces_popular'] = df['province'].copy()
            df.loc[~df['provinces_popular'].isin(provinces.keys().tolist())] = np.nan
            
            self.med_price_by_province = df.groupby(['provinces_popular'], 
                                                     as_index=False).agg({'price':'median'}).\
                                            rename(columns={'price':'MedPriceByProvince',
                                                           'provinces_popular': 'province'})            
        ## 4. Regions    
        if y is not None:
            df['price'] = y.values            
            df['region_1_popular'] = df['region_1'].copy()
            df['region_2_popular'] = df['region_2'].copy()
            df.loc[~df['region_1_popular'].isin(region_1.keys().tolist())] = np.nan
            df.loc[~df['region_2_popular'].isin(region_2.keys().tolist())] = np.nan
            
            self.med_price_by_region = df.groupby(['region_1_popular', 'region_2_popular'], 
                                                     as_index=False).agg({'price':'median'}).\
                                            rename(columns={'price':'MedPriceByRegion',
                                                           'region_1_popular': 'region_1',
                                                           'region_2_popular': 'region_2'})            
        ## 5. Varieties    
        if y is not None:
            df['price'] = y.values            
            df['varieties_popular'] = df['variety'].copy()
            df.loc[~df['varieties_popular'].isin(varieties.keys().tolist())] = np.nan
            
            self.med_price_by_variety = df.groupby(['varieties_popular'], 
                                                     as_index=False).agg({'price':'median'}).\
                                            rename(columns={'price':'MedPriceByVariety',
                                                           'varieties_popular': 'variety'})            
        ## 6. Wineries   
        if y is not None:
            df['price'] = y.values            
            df['wineries_popular'] = df['winery'].copy()
            df.loc[~df['wineries_popular'].isin(wineries.keys().tolist())] = np.nan
            
            self.med_price_by_winery = df.groupby(['wineries_popular'], 
                                                     as_index=False).agg({'price':'median'}).\
                                            rename(columns={'price':'MedPriceByWinery',
                                                           'wineries_popular': 'winery'})            
    
    def transform(self, X):       
        
        # Target encoding
        if self.med_price_by_country is not None:
            X = X.merge(self.med_price_by_country, on=['country'], how='left')
            
        if self.med_price_by_designation is not None:
            X = X.merge(self.med_price_by_designation, on=['designation'], how='left')
        
        if self.med_price_by_province is not None:
            X = X.merge(self.med_price_by_province, on=['province'], how='left')
            
        if self.med_price_by_region is not None:
            X = X.merge(self.med_price_by_region, on=['region_1', 'region_2'], how='left')
            
        if self.med_price_by_variety is not None:
            X = X.merge(self.med_price_by_variety, on=['variety'], how='left')
        
        if self.med_price_by_winery is not None:
            X = X.merge(self.med_price_by_winery, on=['winery'], how='left')
            
        
        # More categorical features
        X = self.points_group(X)        
        
        # Dummy variables
        X = self.get_dummies(X)        
        
        # PCA variables
        X = self.pca_features(X)        
        
        return X
            
    @staticmethod
    def points_group(X):        
        X['points_group'] = np.nan        
        X.loc[X['points'] <= 86, 'points_group'] = 1  
        X.loc[(X['points'] >= 87) & (X['points'] <= 88), 'points_group'] = 2
        X.loc[(X['points'] >= 89) & (X['points'] <= 90), 'points_group'] = 3
        X.loc[(X['points'] >= 91) & (X['points'] <= 95), 'points_group'] = 4
        X.loc[X['points'] >= 96, 'points_group'] = 5
        
        return X
    
    @staticmethod    
    def get_dummies(X):
        ## Points            
        a = pd.get_dummies(X.points_group).rename(columns={1: 'PG_1', 2: 'PG_2', 3: 'PG_3', 4: 'PG_4', 5: 'PG_5'})
        X = pd.concat([X, a], axis=1)       
    
  
        ## clustered 'MedPriceByRegion', 'MedPriceByVariety', 'MedPriceByWinery'
        scaler = StandardScaler()
        a = X[['MedPriceByRegion', 'MedPriceByVariety', 'MedPriceByWinery']].fillna(0)
        a = scaler.fit_transform(a)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(a)
        X.loc[:, 'Cluster_composit'] = kmeans.labels_
        
        a = pd.get_dummies(X.Cluster_composit).rename(columns={0: 'Cluster_1', 1: 'Cluster_2'})
        X = pd.concat([X, a], axis=1).drop('Cluster_composit', axis=1)
        
        return X
    
    @staticmethod   
    def pca_features(X):
        scaler = StandardScaler()
        a = X[['MedPriceByCountry', 'MedPriceByProvince', 'MedPriceByRegion', 
               'MedPriceByVariety', 'MedPriceByWinery']].fillna(0)
        
        a = scaler.fit_transform(a)        
        pca = PCA(n_components=3)
        pca.fit(a)        
        X.loc[:, 'pca_composit_1'] = np.dot(a, pca.components_.T)[:, 0]
        X.loc[:, 'pca_composit_2'] = np.dot(a, pca.components_.T)[:, 1]
        X.loc[:, 'pca_composit_3'] = np.dot(a, pca.components_.T)[:, 2]        
    
        return X

