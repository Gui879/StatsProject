﻿import pandas as pd
import numpy as np
import re
import seaborn as sb
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo
from scipy.stats import kstest
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

final_df=pd.read_csv('final_df.csv')

nans=np.sum(final_df == 0, axis = 0)/final_df.shape[0]

imputs=[np.median(final_df.gini_index),np.mean(final_df.gini_index)]
imputs=[np.random.choice(imputs) + np.random.uniform(0,2)- np.random.uniform(0,2) for _ in range(len(final_df[final_df.gini_index==0]))]
#imputing gini index
final_df.gini_index[final_df.gini_index==0]=imputs

#correcting divorce type:
temp=[float(re.sub(',','.',re.sub('[^\w.,]','',final_df['div_per_100_marriges_'].loc[ix]))) if 
      len(re.findall('\w',str(final_df['div_per_100_marriges_'].loc[ix]))) > 0 else 0 for ix in final_df.index]

final_df['div_per_100_marriges_']=temp


#imputing other variables:
for var in final_df:
    n=len(final_df[final_df[var]==0])
    if n!=0:
        if kstest(final_df[var].values,'norm',N=len(final_df))[1]>0.05:
            change=[np.mean(final_df[var])]*n
            final_df[var][final_df[var]==0]=change
           
        else: 
            change=[np.median(final_df[var])]*n
            final_df[var][final_df[var]==0]=change
            
min_year=final_df.groupby('year')['All ages (%)'].mean().idxmin()
max_year=final_df.groupby('year')['All ages (%)'].mean().idxmax()

max_=final_df[final_df.year==max_year]
min_=final_df[final_df.year==min_year]

overall_corr=final_df.drop(columns=['year','country']).corr()
min_corr=min_.drop(columns=['year','country']).corr()
max_corr=max_.drop(columns=['year','country']).corr()

final_df.drop(columns='health_expenses',inplace=True)
# =============================================================================
# 
# FACTOR ANALYSIS:
# 
# =============================================================================
# =============================================================================
# 
# #seeing if the dataset is suitable for FA:
# kmo_all,kmo_model=calculate_kmo(max_.drop(columns=['year','country']))
# print('max: ',kmo_model)
# 
# kmo_all,kmo_model=calculate_kmo(min_.drop(columns=['year','country']))
# print('min: ',kmo_model)
# 
# #calculating Bartlett’s Test:
# print(calculate_bartlett_sphericity(max_.drop(columns=['year','country'])))
# print(calculate_bartlett_sphericity(min_.drop(columns=['year','country'])))
# 
# fa=FactorAnalyzer(rotation='varimax',n_factors=len(min_.columns))
# 
# #min year:
# fa.fit(min_.drop(columns=['year','country']))
# ev_min,v=fa.get_eigenvalues()
# ev_min=pd.Series(ev_min)
# ev_min.index = range(1,len(ev_min)+1)
# 
# #max year:
# fa.fit(max_.drop(columns=['year','country']))
# ev_max,v=fa.get_eigenvalues()
# ev_max=pd.Series(ev_max)
# ev_max.index = range(1,len(ev_max)+1)
# 
# 
# #num factors min
# sb.lineplot(x=ev_min.index,y=ev_min.values)
# 
# #num factors max
# sb.lineplot(x=ev_max.index,y=ev_max.values)
# 
# =============================================================================


#ENCODING DOS DADOS:
# =============================================================================
# ordinal_df=final_df.copy()
# 
# quartils=pd.DataFrame()
# 
# for year_ in np.unique(final_df.year):
#     for var in final_df.drop(columns=['year','country']):
#         series=final_df[var][final_df.year==year_]
#         Q1=np.percentile(series, 25)
#         Q3=np.percentile(series, 75)
#         quartils=quartils.append({'year':year_,'var':var,'Q1':Q1,'Q3':Q3},ignore_index=True)
#         for row in series.index:
#             if ordinal_df.at[row,var]>=Q3:
#                 ordinal_df.at[row,var]=1
#             elif ordinal_df.at[row,var]<=Q1:
#                 ordinal_df.at[row,var]=-1
#             else: 
#                 ordinal_df.at[row,var]=0
# =============================================================================
    
# =============================================================================
#     
# 
#LINEAR REGRESSION:               
#     
# =============================================================================
df=final_df.copy()
df.drop(columns=['20-24 years old (%)','10-14 years old (%)','70+ years old (%)','30-34 years old (%)','15-19 years old (%)'
                         ,'25-29 years old (%)','50-69 years old (%)','Age-standardized (%)','15-49 years old (%)'],inplace=True)
    
VIS=list(df.drop(columns=['year','country','All ages (%)']).columns)
VD='All ages (%)'

#creating total population :
total=df.iloc[:,12]
for i in range(13,18): total=total + df.iloc[:,i]
df['total']=total

#joining into the following age groups: 0-24,25-64,64+
df['ageGroup_0-24']=df.iloc[:,12]+df.iloc[:,13]
df['ageGroup_25-64']=df.iloc[:,14]+df.iloc[:,15]
df['ageGroup_65_more']=df.iloc[:,16]+df.iloc[:,17]

#turning the age groups into percentage:
df['ageGroup_0-24']=df['ageGroup_0-24']/df['total']
df['ageGroup_25-64']=df['ageGroup_25-64']/df['total']
df['ageGroup_65_more']=df['ageGroup_65_more']/df['total']

#dropping unecessary columns
drop_col=[df.iloc[:,i].name for i in range(12,18)]
drop_col.append('total')
df.drop(columns=drop_col,inplace=True)

df['year']=df['year'].map(lambda x: int(str(x)[-2::])+1)
df.rename(columns={'All ages (%)':'depression_rate','div_per_100_marriges_':'divorce_p_100_marriges','Per capita CO₂ emissions (Global Carbon Project; Gapminder; UN) (tonnes per capita)':'CO2_emissions'},inplace=True)


df.to_csv('ready_df')
# =============================================================================
# 
# 
# import seaborn as sns; sns.set()
# 
# # Generate a mask for the upper triangle
# mask = np.zeros_like(corr, dtype=np.bool)
# mask[np.triu_indices_from(mask)] = True
# 
# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(11, 9))
# 
# # Generate a custom diverging colormap
# cmap = sb.diverging_palette(220, 10, as_cmap=True)
# 
# # Draw the heatmap with the mask and correct aspect ratio
# sns_plot=sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5})
# 
# sns_plot.figure.savefig('visualizations/'+'overall'+'_corr'+".png")
# plt.clf()
# 
# 
# 
# 
# # =============================================================================
# # 
# # ASSUMPTIONS OF LINEAR REGRESSION:
# # 
# # =============================================================================
# import statsmodels.api as sm
# from matplotlib import pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from scipy.stats import boxcox
# from scipy.stats import shapiro  as shap
# 
# data=df.copy()
# data['All ages (%)'] = sm.add_constant(data['All ages (%)'])
# mod_fit = sm.OLS(data['All ages (%)'], data['All ages (%)']).fit()
# res = mod_fit.resid # residuals
# fig = sm.qqplot(res)
# plt.show()
# 
# #cube root transformation
# sb.distplot(df['All ages (%)'] ** (1. / 3))
# #Square transformation
# sb.distplot(df['All ages (%)'] ** 2)
# #log transformation:
# sb.distplot(np.log(df['All ages (%)']))
# #square root transformation:
# sb.distplot(np.sqrt(df['All ages (%)']))
# # min max scaling
# scaler = MinMaxScaler()
# fitted=scaler.fit_transform(df['All ages (%)'].values.reshape(-1, 1))
# sb.distplot(fitted)
# #boxcox of the exponential
# sb.distplot(boxcox(np.exp(df['All ages (%)']),0))
# 
# #exponential transformation:
# sb.distplot(np.exp(df['All ages (%)']))
# 
# #shapiro wilk test:
# shap(np.sqrt(df['All ages (%)']))[1]<0.05
# 
# for year_ in np.unique(final_df.year):
#     sns_plot = sns.scatterplot(x=var, y=dependent_variable, data=dt)
#     sns_plot.figure.savefig('scatters/'+dependent_variable+'_'+var+'_'+str(year_)+".png")
#     plt.clf()
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# import matplotlib.pyplot as plt
# 
# numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
# 
# independent_variables = list(df.select_dtypes(include=numerics).columns)
# independent_variables.remove('All ages (%)')
# dependent_variable = 'All ages (%)'
# 
# for var in independent_variables:
#     for year_ in np.unique(final_df.year):
#         dt=df[df.year==year_]
#         sns_plot = sns.scatterplot(x=var, y=dependent_variable, data=dt)
#         sns_plot.figure.savefig('scatters/'+dependent_variable+'_'+var+'_'+str(year_)+".png")
#         plt.clf()
# 
# 
# for var in independent_variables:
#     sns_plot = sns.distplot(df[var])
#     sns_plot.figure.savefig('histograms/'+'_'+var+".png")
#     plt.clf()
# 
# for var in independent_variables:
#     sns_plot = sns.scatterplot(data=df[var])
#     sns_plot.figure.savefig('histograms/'+'_'+var+".png")
#     plt.clf()
#     
# 
# 
# 
# 
# 
# 
# 
# models=[]
# summaries=[]
# for year_ in np.unique(df.year):
#    #filtering the year:
#     temp=df[df.year==year_]
#     #defining variables:
#     X=temp.drop(columns=['country','year','All ages (%)'])
#     X=sm.add_constant(X)    
#     Y=temp['All ages (%)']
#     #estimating the model:
#     model=sm.OLS(Y,X).fit()
#     #appending the summary:
#     summaries.append(model)
#     
#     #appending results:
#     temp_results=pd.DataFrame(columns=['coef','pvalue'],index=X.columns)
#     temp_results['coef']=model.params
#     temp_results['pvalue']=model.pvalues
#     models.append(temp_results)
# 
# X=df.drop(columns=['country','year','All ages (%)'])
# Y=df['All ages (%)']
# model=sm.OLS(Y,X).fit()
# model.summary()
# 
# 
# 
# =============================================================================

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

independent_variables = list(df.select_dtypes(include=numerics).columns)
independent_variables.remove('All ages (%)')
dependent_variable = 'All ages (%)'

for var in independent_variables:
    sns_plot = sns.scatterplot(x=var, y=dependent_variable, data=df)
    sns_plot.figure.savefig('scatters/'+dependent_variable+'_'+var+".png")
    plt.clf()


for var in independent_variables:
    sns_plot = sns.distplot(df[var])
    sns_plot.figure.savefig('histograms/'+'_'+var+".png")
    plt.clf()

for var in independent_variables:
    sns_plot = sns.scatterplot(data=df[var])
    sns_plot.figure.savefig('histograms/'+'_'+var+".png")
    plt.clf()
    
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

model = sm.OLS(df['All ages (%)'], df.drop(['All ages (%)','year', 'country'], axis =1))
results = model.fit()
print(results.summary())

models=[]
summaries=[]
for year_ in np.unique(df.year):
   #filtering the year:
    temp=df[df.year==year_]
    #defining variables:
    X=temp.drop(columns=['country','year','All ages (%)'])
    X=sm.add_constant(X)    
    Y=temp['All ages (%)']
    #estimating the model:
    model=sm.OLS(Y,X).fit()
    #appending the summary:
    summaries.append(model)
    
    #appending results:
    temp_results=pd.DataFrame(columns=['coef','pvalue'],index=X.columns)
    temp_results['coef']=model.params
    temp_results['pvalue']=model.pvalues
    models.append(temp_results)

X=df.drop(columns=['country','year','All ages (%)'])
Y=df['All ages (%)']
model=sm.OLS(Y,X).fit()
model.summary()