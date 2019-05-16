#Getting the df ready:
df=read.csv('ready_df')
drop<- 'X'
df<-df[ , !(names(df) %in% drop)]
drop<-'country'
df_scaled<-data.frame(scale(df[ , !(names(df) %in% drop)], center = TRUE, scale = TRUE))


#LINEAR REGRESSION:
model = lm(depression_rate~0+year+education+gdp_per_capita+gini_index+greenhousepc+pop_density+productivity_per_hour+
             remuneration_per_capita+unemployment+inflation+divorce_p_100_marriges+CO2_emissions+ageGroup_0.24+
             ageGroup_25.64+ageGroup_65_more, data=df_scaled)

summary(model)

#LINEAR REGRESSION ASSUMPTIONS:
#1) linear relational nao se ve em pairplot.
#seeing the distribution of our dependent variable

#2) erros sao independentes:



#3) erros com distribuicao normal:



#3.1) variavel dependente com distribuicao normal:
library(MASS)
#truehist(df$depression_rate, main="Distribution of average depression rate")
plot(density(df$depression_rate),main="Distribution of average depression rate")
abline(v=median(df$depression_rate), col="red")
abline(v=mean(df$depression_rate), col="blue")
#making the Q-Q plot
qqnorm(df$depression_rate)
qqline(df$depression_rate,lwd=2,col='red')


#4) multicolinariedade (lol) das variaveis:
#install.packages('corrplot')
library(corrplot)
corrplot(cor(df[ , !(names(df) %in% 'country')]), type="lower")


#ver partial correlation matrix.



