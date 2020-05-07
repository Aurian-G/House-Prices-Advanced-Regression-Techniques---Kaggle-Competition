##Predicting Home Prices Kaggle

library(GGally)
library(naniar)
library(tidyverse)
library(car)
library(caret)
library(leaps)

##Question 1
##We want to know if there is a difference in price by location with respect to square foot? 

#Read in Train set
train = read.csv(file.choose(), header=TRUE, stringsAsFactors = FALSE)

#Check for NA values in Train set
gg_miss_var(train)

#Missing Vars: PoolQC, MiscFeature, Alley, Fence, FireplaceQu, LotFrontage,
#GarageYrBlt/Type/Qual/Finish/Cond, BsmtFinType2/Exposure/Qual/FinType1/Cond, MasVnrType/Area

##Variable transformations and scatterplot Matrix for variables of interest
#SalePrice, GrLivArea(Sqft), Neighborhood(NAmes, Edwards and BrkSide)
#Change GrLivArea to Sqft for readability purposes
names(train)[47] = "Sqft"

#Reset Factor Levels in Train DF for Neighborhood
Ntrain = train %>% filter(Neighborhood == "NAmes" | Neighborhood == "Edwards" | Neighborhood == "BrkSide")
Ntrain$Neighborhood = factor(Ntrain$Neighborhood)

#Histogram Distribution of Sqft
#We see right skewness which can break our normality assumption
Ntrain%>%
  select(Sqft)%>%
  ggplot(aes(x=Sqft))+
  geom_histogram(fill="blue", col = "red")+
  ggtitle("Histogram Distribution of Sqft")

#Histogram Distribution of Sale Price
#More right skewness present
Ntrain%>%
  select(SalePrice)%>%
  ggplot(aes(x=SalePrice))+
  geom_histogram(fill="blue", col = "red")+
  ggtitle("Histogram Distribution of Sale Price")

####Histogram distributions of both variables show a need for transformations 
####due to skewness and clusters. Transformation is necessary to assert normality assumption

#plot and observe if flexible slopes is necessary w/ facet wrap
#Edwards has 2 points that have both high leverage and high residual
Ntrain %>%
  select(SalePrice, Sqft, Neighborhood)%>%
  ggplot(aes(x=Sqft, y=SalePrice, color = Neighborhood))+
  geom_point()+
  facet_wrap(~Neighborhood)

#plot and observe if flexible slopes is necessary w/o facet wrap
Ntrain %>%
  select(SalePrice, Sqft, Neighborhood)%>%
  ggplot(aes(x=Sqft, y=SalePrice, color = Neighborhood))+
  geom_point()+
  geom_smooth(method="lm")

####Scatter plots show some evidence of heteroscedasticity. Confirms our initial claim above
####that transformation is needed.
####Residual plots also show tight clusters around a focal point
##Seem to be quite a few influential points that are breaking our assumptions of equal standard dev
##as well as clusters of points between 1000-2000 which may break our normality assumption
##Log both variables to see if we get a more clean look of data dispersion
#Create log variables
Ntrain$LogSqft = log(Ntrain$Sqft)
Ntrain$LogSalePrice = log(Ntrain$SalePrice)

#Histogram Distribution of LogSqft
Ntrain%>%
  select(LogSqft)%>%
  ggplot(aes(x=LogSqft))+
  geom_histogram(fill="blue", col = "red")+
  ggtitle("Histogram Distribution of Log Sqft")

#Histogram Distribution of LogSalePrice
Ntrain%>%
  select(LogSalePrice)%>%
  ggplot(aes(x=LogSalePrice))+
  geom_histogram(fill="blue", col = "red")+
  ggtitle("Histogram Distribution of Log Sale Price")


#Scatterplot with log variables
Ntrain %>%
  select(LogSalePrice, LogSqft, Neighborhood)%>%
  ggplot(aes(x=LogSqft, y=LogSalePrice, color = Neighborhood))+
  geom_point()+
  facet_wrap(~Neighborhood)

#Scatterplot with log variables and MLR visual
Ntrain %>%
  select(LogSalePrice, LogSqft, Neighborhood)%>%
  ggplot(aes(x=LogSqft, y=LogSalePrice, color = Neighborhood))+
  geom_point()+
  geom_smooth(method="lm")

##Normality assumption looks great with visual evidence of normal distribution both within the
##explanatory variables and residual plots
##Residual plots also show good dispersion of points in a cloud-like manner.
##No overwhelming evidence of differing standard deviations
##Feeling more confident with log transformation. Will move forward with log variables
##Set "NAmes" as reference variable
Ntrain <- within(Ntrain, Neighborhood <- relevel(Neighborhood, ref="NAmes"))

##Visual evidence and QOI suggests to build a full model with flexible slopes
#Adj R-Sq 0.5056. The intercept and slope of Edwards and NAmes don't have enough evidence to suggest they are not equal(p-value = 0.348,0.520,respectively)
summary(lm(LogSalePrice~LogSqft*Neighborhood, data = Ntrain))

#Running a model without interaction to see how they line up
#Adj R-Sq 0.4857. All terms signif
summary(lm(LogSalePrice~LogSqft+Neighborhood, data = Ntrain))

##Lets do test to compare the fits w/ and w/o interaction terms
fit1 = summary(lm(LogSalePrice~LogSqft*Neighborhood, data = Ntrain))
fit2 = summary(lm(LogSalePrice~LogSqft+Neighborhood, data = Ntrain))

##Full Model
#MSE fit1 = 0.0364
#SSR fit1 = 13.938
#DF 377
mean(fit1$residuals^2)
sum(fit1$residuals^2)

##Reduced Model
#MSE fit2 = 0.0381
#SSR fit2 = 14.577
#DF 379
mean(fit2$residuals^2)
sum(fit2$residuals^2)

#After constructing ANOVA
#DF1 = 2, DF2 = 377, F-Stat = 8.78
pf(8.78,2,377,lower.tail = FALSE)

#There is strong evidence to suggest that at least 1 of the Neighborhood groups have different
#slopes than the others (p-value: 0.00019). We can proceed with interactive variable model.

##Residual Plots show 2 influential points on Cook's D that show high leverage and residual
##Investigate the 2 influential points for Edwards Neighborhood
Ntrain %>%
  select(LogSqft, Id)%>%
  arrange(LogSqft)

#We can restrict the range of the model to exclude the 2 outlying observations
#Can filter to show only results < 8.3, or median sqft < 4,000
#Reasoning for this is because a majority of the data (381 obs) is in the range <4,000
#while there is only 2 obs in the range >4,000
#These outliers may explain the appearance of collinearity we saw between NAmes and Edwards
Ntrain2 = Ntrain[-c(131,339),]

#Model without observation present
#0.02 change in Adjusted R-Square. 0.50 -> 0.52
#Very important to note. NAmes and BrkSide intercept and slope now have enough evidence 
#to suggest they both are different(p-value:.008,.015,respectively). This means the 2 outliers
#had an associated effect of making the two neighborhoods seems to be collinear. The correlation
#and redundancy of those 2 neighborhoods can be explained by the outliers
fit3 = summary(lm(LogSalePrice~LogSqft*Neighborhood, data = Ntrain2))
fit3

#Question of Interest
#The client would like an estiamte of how the Sales Price relates to Square footage while 
#taking into consideration the effect of Neighborhood location (NAmes, Edwards and BrkSide).

#Fit Model 3 (restricted range and inclusion of interactive terms) is best model
fit3
#b0 = 8.49
#b1 = 0.47logsqft
#b2 = -2.58BrkSide
#b3 =  -1.57Edwards
#b4 = 0.35logsqft*BrkSide
#b5 = 0.20logsqft*Edwards
#Full Model: 8.49+0.47logsqft-2.58BrkSide-1.57Edwards+(0.35logsqft*BrkSide)+(0.20logsqft*Edwards)



###Variable Selection Distributions and count statistics for variables 60-81 (22 total)
#filter all rows past 60
Ntrain4 = train[,60:81]

#GarageYrBuilt
#Left Skewed, Most houses built around 1990-2010
Ntrain4 %>% ggplot(aes(x=GarageYrBlt)) + geom_histogram(fill="blue", col = "red")

#GarageFinish
#Unf = 605, RFn = 422, Fin = 352, NA = 81
Ntrain4 %>% ggplot(aes(x=GarageFinish)) + geom_bar(fill="blue", col = "red") +geom_text(stat='count', aes(label=..count..), vjust=-0.5)

#GarageCars
#0 = 81, 1 = 369, 2 = 824, 3 = 181, 4 = 5
Ntrain4 %>% ggplot(aes(x=GarageCars)) + geom_bar(fill="blue", col = "red") +geom_text(stat='count', aes(label=..count..), vjust=-0.5)

#GarageArea
#Slight Right skew. Median seems to be better unit of measure with the median centered around 500
Ntrain4 %>% ggplot(aes(x=GarageArea)) + geom_histogram(fill="blue", col = "red")

#GarageQual
#Ex = 3, Fa = 48, Gd = 14, TA = 1311, NA = 81
Ntrain4 %>% ggplot(aes(x=GarageQual)) + geom_bar(fill="blue", col = "red") +geom_text(stat='count', aes(label=..count..), vjust=-0.5)

#GarageCond
#Ex = 2, Fa = 35, Gd = 9, Po = 7, TA = 1326, NA = 81
Ntrain4 %>% ggplot(aes(x=GarageCond)) + geom_bar(fill="blue", col = "red") +geom_text(stat='count', aes(label=..count..), vjust=-0.5)

#PavedDrive
#Y = 1340, N = 90, P = 30
Ntrain4 %>% ggplot(aes(x=PavedDrive)) + geom_bar(fill="blue", col = "red") +geom_text(stat='count', aes(label=..count..), vjust=-0.5)

#WooddeckSF
#VERY Right Skewed. Heavely centered at 0
Ntrain4 %>% ggplot(aes(x=WoodDeckSF)) + geom_histogram(fill="blue", col = "red")

#OpenPorch
#VERY Right Skewed. Heavely centered at 0
Ntrain4 %>% ggplot(aes(x=OpenPorchSF)) + geom_histogram(fill="blue", col = "red")

#Enclosed Porch
#VERY Right Skewed. Heavely centered at 0. 
Ntrain4 %>% ggplot(aes(x=EnclosedPorch)) + geom_histogram(fill="blue", col = "red")

#X3SsnPorch
#VERY Right Skewed. Heavely centered at 0. Only 2 observations not centered at 0
Ntrain4 %>% ggplot(aes(x=X3SsnPorch)) + geom_histogram(fill="blue", col = "red")

#Screenporch
#VERY Right Skewed. Heavely centered at 0. Less than 50 obs are nonzero. May leave out
Ntrain4 %>% ggplot(aes(x=ScreenPorch)) + geom_histogram(fill="blue", col = "red")

#PoolArea
#VERY Right Skewed. Heavely centered at 0. Basically all obs are at zero here except 3 obs.
Ntrain4 %>% ggplot(aes(x=PoolArea)) + geom_histogram(fill="blue", col = "red")

#PoolQC
#Ex = 2, Fa = 2, Gd = 3, NA = 1453. Most are NA, exclude this Variable
Ntrain4 %>% ggplot(aes(x=PoolQC)) + geom_bar(fill="blue", col = "red") +geom_text(stat='count', aes(label=..count..), vjust=-0.5)

#Fence
#GdPrv = 59, GdWo = 54, MnPrv = 157, MnWw = 11, NA = 1179. A lot of NA, may want to exclude variable
Ntrain4 %>% ggplot(aes(x=Fence)) + geom_bar(fill="blue", col = "red") +geom_text(stat='count', aes(label=..count..), vjust=-0.5)

#MiscFeature
#Gar2 = 2, Othr = 2, Shed = 49, TenC = 1, NA = 1406
Ntrain4 %>% ggplot(aes(x=MiscFeature)) + geom_bar(fill="blue", col = "red") +geom_text(stat='count', aes(label=..count..), vjust=-0.5)

#MiscVal
#VERY Right Skewed. Heavely centered at 0. Can leave out variable, only few non-zero obs.
Ntrain4 %>% ggplot(aes(x=MiscVal)) + geom_histogram(fill="blue", col = "red")

#MoSold
#Normal Distribution centered between 5-7.5
Ntrain4 %>% ggplot(aes(x=MoSold)) + geom_histogram(fill="blue", col = "red")

#YrSold
#Uniform Distibution. Relatively even # of units sold between all years 2006-2009. 2010 has least
Ntrain4 %>% ggplot(aes(x=YrSold)) + geom_histogram(fill="blue", col = "red")

#SaleType
#COD = 43, Con = 2, ConLD = 9, ConLI/LW = 5/5, CWD = 4, New = 122, Oth = 3, WD = 1267
Ntrain4 %>% ggplot(aes(x=SaleType)) + geom_bar(fill="blue", col = "red") +geom_text(stat='count', aes(label=..count..), vjust=-0.5)

#SaleCondition
#Abnormal = 101, AdjLand = 4, Alloca = 12, Family = 20, Normal = 1198, Partial = 125
Ntrain4 %>% ggplot(aes(x=SaleCondition)) + geom_bar(fill="blue", col = "red") +geom_text(stat='count', aes(label=..count..), vjust=-0.5)

#SalePrice
#Right skewed. Median centered around 150-200k
Ntrain4 %>% ggplot(aes(x=SalePrice)) + geom_histogram(fill="blue", col = "red")

##Scatterplot Matrix of all Continuous variables
##Left out PoolQC, PoolArea, ScreenPorch, and Misc Val since 98% of each variable had obs = 0
pairs(~SalePrice + GarageYrBlt + GarageArea + WoodDeckSF + OpenPorchSF + EnclosedPorch + MoSold + YrSold, data = Ntrain4)

##We have weak visual positive linear correlation with SalePrice vs GarageYrBlt, GarageArea, WoodDeck and Open Porch.
##Will log SalePrice to see if it fixes our plots
Ntrain4$LogSalePrice = log(Ntrain4$SalePrice)

##LogSalePrice Plots
pairs(~LogSalePrice + GarageYrBlt + GarageArea + WoodDeckSF + OpenPorchSF + EnclosedPorch + MoSold + YrSold, data = Ntrain4)

#Plots show strong visual relationship of positive linear relationship with GarageYrBlt, GarageArea, WoodDeckSF, and MAYBE with OpenPorchSF/EnclosedPorch.
#Mo and YrSold seem to have no linear relationship with LogSalePrice



####VARIABLE SELECTION

##Fit the full Model
##Error showed that we need to relevel factors through some data cleaning to remove NA's
full_model = lm(LogSalePrice~., data = train)
str(train)

#Drop factor variables with less than 2 levels & keep non-factor vars
#Refactored Code via dmi3kno @ Kaggle
names(train) <- make.names(names(train))

features <- setdiff(colnames(train), c("Id", "SalePrice"))
for (f in features) {
  if (any(is.na(train[[f]]))) 
    if (is.character(train[[f]])){ 
      train[[f]][is.na(train[[f]])] <- "Others"
    }else{
      train[[f]][is.na(train[[f]])] <- -999  
    }
}

column_class <- lapply(train,class)
column_class <- column_class[column_class != "factor"]
factor_levels <- lapply(train, nlevels)
factor_levels <- factor_levels[factor_levels > 1]
train <- train[,names(train) %in% c(names(factor_levels), names(column_class))]

train <- as.data.frame(unclass(train))

#Filter out SalePrice data and keep only LogSalePrice so it is not being used as a predictor in the code below
train = subset(train, select = -SalePrice)

#### STEPWISE REGRESSION MODEL
#Build the full model
full.model <- lm(LogSalePrice ~., data = train)

#Stepwise regression model
step.model <- stepAIC(full.model, direction = "both", trace = FALSE)
summary(step.model)

###Backward Selection
step.model <- stepAIC(full.model, direction = "backward", trace = FALSE)
summary(step.model)

###Forward Selection
step.model <- stepAIC(full.model, direction = "forward", trace = FALSE)
summary(step.model)
