# -*- coding: utf-8 -*-
"""
Created on Sat May 11 13:58:56 2019

@author: Guilherme
"""

import pandas as pd
import re
import pandas_profiling as pp
import numpy as np

### Data Loading
health_expenses = pd.read_excel('desp_saude.xlsx',sheet_name = 'Folha1',header= 1)
education = pd.read_excel('education.xlsx',sheet_name = 'Folha1',header= 1)
gdp_per_capita = pd.read_excel('gdp_per_capita.xlsx',sheet_name = 'Folha1',header= 1)
gini_index = pd.read_excel('gini_index.xlsx',sheet_name = 'Folha1',header= 1)
greenhousepc = pd.read_excel('greenhousepc.xlsx',sheet_name = 'Folha1',header= 1)
pop_density = pd.read_excel('pop_density.xlsx',sheet_name = 'Folha1',header= 1)
pop_poverty_risk = pd.read_excel('pop_risco_pobreza.xlsx',sheet_name = 'Folha1',header= 1)
precipitation = pd.read_excel('precipitation.xlsx', sheet_name = 'Folha1', header= 1)
inflation = pd.read_excel('inflation_CPI.xlsx', sheet_name = 'Folha1', header= 1)
div_per_100_marriges_ = pd.read_excel('div_per_100_marriges_.xlsx', sheet_name = 'Folha1', header= 1)

population_struct = pd.read_excel('estrutura_populacao.xlsx', sheet_name = 'Folha1', header = 1)
#Splitting pop struct by age group
age_groups = ['0-14','15-24','25-49','50-64','65-79','80-more']
pop_l = []
for i in range(66, len(population_struct.columns)+1, 33):
    columns =population_struct.columns[i-33:i]
    temp = population_struct[columns]
    temp.columns = [re.sub('\..*','',col) for col in columns]
    pop_l.append(temp)

productivity_per_hour = pd.read_excel('produtividade_hora_trabalho.xlsx',sheet_name = 'Folha1',header= 1)
remuneration_per_capita = pd.read_excel('remuneracao_per_capita.xlsx', sheet_name = 'Folha1', header = 1)
unemployment = pd.read_excel('taxa_desemprego.xlsx', sheet_name = 'Folha1', header = 1)

land_area = pd.read_csv('land-area-km.csv')

co_emission_per_capita = pd.read_csv('co-emissions-per-capita.csv')
co_emission_per_capita = co_emission_per_capita.rename(index=str, columns={"Year": "year", "Entity": "country"})
#Transforming co emission
co_emission_per_capita.drop('Code', axis = 1, inplace = True)
co_emission_per_capita.set_index(['year','country'],inplace =True, drop = True)

prevalence_of_depression = pd.read_csv('prevalence-of-depression-by-age.csv')
prevalence_of_depression = prevalence_of_depression.rename(index=str, columns={"Year": "year", "Entity": "country"})
#Transforming depression data
prevalence_of_depression.drop('Code', axis = 1, inplace = True)
prevalence_of_depression.set_index(['year','country'],inplace =True, drop = True)

variables = [health_expenses,education,gdp_per_capita,gini_index,greenhousepc,pop_density, pop_poverty_risk, 
             productivity_per_hour,remuneration_per_capita,unemployment,co_emission_per_capita] 

variables_with_no_poppov = [health_expenses,education,gdp_per_capita,gini_index,greenhousepc,pop_density, 
             productivity_per_hour,remuneration_per_capita,unemployment,inflation,div_per_100_marriges_]+ pop_l
                            
st = 'health_expenses,education,gdp_per_capita,gini_index,greenhousepc,pop_density,productivity_per_hour,remuneration_per_capita,unemployment,inflation,div_per_100_marriges_,'
st = st + ','.join(age_groups)
var_names = st.split(',')

def transform_variables(variables):
    dfs = []
    for ix,var in enumerate(variables):
        try:
            var = var.drop(['EU28 - European Union (28 countries)', 'EA19 - Euro Area (19 countries)'], axis =1)
        except:
            pass
        shape = var.shape
        col = []
        for i in range(shape[1]):
            for j in range(shape[0]):
                year = var.index[j]
                country = var.columns[i]
                country = re.sub(r'[A-Z]+ - ','',country)

                col.append((str(year),str(country),var.iloc[j,i]))
        temp = pd.DataFrame(col,columns = ['year','country',var_names[ix]])
        temp['year'] = temp.year.astype('int64')
        temp.set_index(['year','country'],inplace = True, drop = True)
        dfs.append(temp)
    return dfs
      
dfs = transform_variables(variables_with_no_poppov)
final_df = dfs[0]
for i in range(1,len(dfs)):
    final_df = final_df.join(dfs[i], how = 'inner')

nonpor = co_emission_per_capita.join(prevalence_of_depression, how = 'inner')
final_df = final_df.join(nonpor, how = 'inner')
final_df.to_csv('final_df.csv')