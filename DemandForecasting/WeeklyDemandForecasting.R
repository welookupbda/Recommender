# Written by Sai Krishnan Mohan

setwd("/home/sai/Downloads/Retail_Data")

#Read in the file
#Need to include dataset from Srini
product_csv      <- read.csv("product_data.csv",      header=T,stringsAsFactors = FALSE)
#transaction_csv  <- read.csv("transaction_data.csv", header=T,stringsAsFactors = FALSE)

#Create copies
product_data     <- product_csv
#transaction_data <- transaction_csv

colnames(product_data)
#colnames(transaction_csv)

#Extract data for Malhar Mega Mall only
product_data <- product_data[product_data$store_description=="MM-INDORE-MALHAR MEGA MALL",]

#Convert transaction date to weekly format
product_data$transactionDate <- as.Date(product_data$transactionDate, "%Y-%m-%d")

#Extract only product code, description and date fields
library(dplyr)

product_data_for_agg <- product_data %>% 
                        select(transactionDate, product_code, product_description)

product_data_agg     <- product_data_for_agg %>% 
                        group_by(transactionDate, product_code) %>% 
                        mutate(count = n())

#Select only 2017 data
library(lubridate)
product_data_agg_2017 <- product_data_agg[year(product_data_agg$transactionDate) == 2017,]

#Exclude set of products
product_data_agg_2017_filtered <- product_data_agg_2017 %>% 
                                  filter(!product_description %in% c('ONION LOOSE','POTATO LOOSE','TOMATO LOOSE','CORIANDER', 
                                         'CUCUMBER GREEN LOOSE', 'LADYFINGER LOOSE','BOTTLE GOURD LONG','CABBAGE',
                                         'CAPSICUM GREEN','Carrot English Loose', 'GINGER', 'LEMON LOOSE',
                                         'SUGAR MEDIUM LOOSE','TATA SALT PP 1Kg','CAULIFLOWER', 'FB VG CHILLI RED MIRCHI',
                                         'BRINJAL BHARTA PURPLE', 'METHI','RIDGE GOURD', 'FB SIS NAMKEENS',
                                         'BB-CB-20X25X168SWG-Suitable for ROI New','BB-CB-20X25X208SWG NEW','BB-CB-27X30X208SWG NEW',
                                         'BANANA ROBUSTA RAW,BB-CB-27X30X168SWG-Suitable for ROI New','GARLIC PREMIUM','GREEN PEAS',
                                         'BANANA ROBUSTA RAW','BB-CB-27X30X168SWG-Suitable for ROI New','BROCCOLI','CAPSICUM RED LOOSE',
                                         'MUSHROOM BUTTON'))

#Aggregate data to weekly
product_data_agg_2017_filtered$Week <- week(product_data_agg_2017_filtered$transactionDate)
product_data_2017_filtered_weekly   <- aggregate(count~Week+product_code+product_description, FUN=sum, data=product_data_agg_2017_filtered, na.rm=TRUE)
product_data_2017_filtered_weekly   <- product_data_2017_filtered_weekly[order(product_data_2017_filtered_weekly$Week, -product_data_2017_filtered_weekly$count),]
str(product_data_2017_filtered_weekly)

#Select products for modeling demand based on maximum demand over 2017
library(dplyr)
topn_products <- product_data_2017_filtered_weekly %>% 
                 group_by(product_code, product_description) %>% 
                 summarise(Total = sum(count)) %>%
                 arrange(-Total) %>%
                 top_n(n = 3, wt = Total)

#We are selecting MAGGI NDL MASALA 420g and WATERMELON KIRAN to prepare demand forecasts based on time series
product_data_agg_2017_maggi <- product_data_2017_filtered_weekly %>% 
                               filter(product_description %in% c('MAGGI NDL MASALA 420g'))
product_data_agg_2017_watermelonKiran <- product_data_2017_filtered_weekly %>% 
                               filter(product_description %in% c('WATERMELON KIRAN'))

#Create time series datasets for ARIMA modeling
product_data_agg_2017_maggi_ts       <- ts(product_data_agg_2017_maggi$count)
product_data_agg_2017_watermelonKiran_ts <- ts(product_data_agg_2017_watermelonKiran$count)

class(product_data_agg_2017_maggi_ts)
class(product_data_agg_2017_watermelonKiran_ts)

#Developing ARIMA model for maggi
frequency(product_data_agg_2017_maggi_ts)
summary(product_data_agg_2017_maggi_ts)
plot.ts(product_data_agg_2017_maggi_ts)
abline(reg=lm(product_data_agg_2017_maggi_ts~time(product_data_agg_2017_maggi_ts)))
cycle(product_data_agg_2017_maggi_ts)

boxplot(product_data_agg_2017_maggi_ts~cycle(product_data_agg_2017_maggi_ts))

library(tseries)
adf.test(diff(log(product_data_agg_2017_maggi_ts)), alternative="stationary", k=0)

#This is a stationary time series
#Dickey-Fuller = -5.4639, Lag order = 0, p-value = 0.01
#alternative hypothesis: stationary

#ACF plot
acf(product_data_agg_2017_maggi_ts)

library(forecast)
set.seed(1234)
fit <- auto.arima(product_data_agg_2017_maggi_ts[1:21])
arimaorder(fit) #p, d, q are 0, 0, 0

pred_maggi <- predict(fit, n.ahead = 4)
pred_maggi #Gives a steady demand of 1300 units per week

#Checking for MAPE for maggi
library(MLmetrics)
MAPE(pred_maggi$pred,product_data_agg_2017_maggi_ts[22:25])
#MAPE is 0.5169

#Fit exponential smoothing model
library(expsmooth)
#############
beta      <- seq(.0001, .5, by = .001)
RMSE      <- NA
for(i in seq_along(beta)) {
  fit     <- holt(product_data_agg_2017_maggi_ts[1:21], beta = beta[i], h = 4)
  RMSE[i] <- accuracy(fit, product_data_agg_2017_maggi_ts[22:25])[2,2]
}

# convert to a data frame and idenitify min alpha value
beta.fit <- data_frame(beta, RMSE)
beta.min <- filter(beta.fit, RMSE == min(RMSE))

# plot RMSE vs. alpha
library(ggplot2)
ggplot(beta.fit, aes(beta, RMSE)) +
  geom_line() +
  geom_point(data = beta.min, aes(beta, RMSE), size = 2, color = "blue")  

#We get optimum value of beta around 0.126. Building holts model around this.
maggi_holt <- holt(product_data_agg_2017_maggi_ts[1:21], beta = 0.126, h = 4)
MAPE(maggi_holt$mean, product_data_agg_2017_maggi_ts[22:25])
#MAPE with Holts model is 0.4289

#Building Holts winters model with Beta =0.126
maggi.smooth<- HoltWinters(product_data_agg_2017_maggi_ts[1:21], beta=0.4289, gamma=FALSE)
plot(maggi.smooth$fitted)
maggi.smooth_pred <- forecast(maggi.smooth, h=4)
accuracy(maggi.smooth_pred,product_data_agg_2017_maggi_ts[22:25])
#MAPE from Holts Winters on test set is 50.57474

#Finally, let's try predicting through moving average and see if we getter a better MAPE
library(pracma)
movAvgsimple       <- movavg(product_data_agg_2017_maggi_ts[1:24], 3, "s")
predmovAvgsimple   <- (product_data_agg_2017_maggi_ts[24] + product_data_agg_2017_maggi_ts[23] + product_data_agg_2017_maggi_ts[22]) / 3
MAPE(predmovAvgsimple, product_data_agg_2017_maggi_ts[25])
# MAPE is 0.0522. This is good. Taking this a step back and predicting the 24th week demand.

movAvgsimple       <- movavg(product_data_agg_2017_maggi_ts[1:23], 3, "s")
predmovAvgsimple   <- (product_data_agg_2017_maggi_ts[23] + product_data_agg_2017_maggi_ts[22] + product_data_agg_2017_maggi_ts[21]) / 3
MAPE(predmovAvgsimple, product_data_agg_2017_maggi_ts[24])
# MAPE is 1.847

#Best results for predicting maggi weekly demand are from Simple moving average and Holt's model
#############

#############
#Developing ARIMA model for Watermelon Kiran
frequency(product_data_agg_2017_watermelonKiran_ts)
summary(product_data_agg_2017_watermelonKiran_ts)
plot(product_data_agg_2017_watermelonKiran_ts)
abline(reg=lm(product_data_agg_2017_watermelonKiran_ts~time(product_data_agg_2017_watermelonKiran_ts)))
cycle(product_data_agg_2017_watermelonKiran_ts)

boxplot(product_data_agg_2017_watermelonKiran_ts~cycle(product_data_agg_2017_watermelonKiran_ts))

library(tseries)
adf.test(diff(log(product_data_agg_2017_watermelonKiran_ts)), alternative="stationary", k=0)

#This is a stationary time series
#Dickey-Fuller = -5.8093, Lag order = 0, p-value = 0.01
#alternative hypothesis: stationary

#ACF plot
acf(product_data_agg_2017_watermelonKiran_ts)

library(forecast)
set.seed(1234)
fit1 <- auto.arima(product_data_agg_2017_watermelonKiran_ts[1:22])
arimaorder(fit1) #p, d, q are 0, , 0

pred_watermelonKiran <- predict(fit1, n.ahead = 4)
pred_watermelonKiran #Gives a steady demand of 21494 units per week.

MAPE(pred_watermelonKiran$pred,product_data_agg_2017_watermelonKiran_ts[23:26]) 
#MAPE is 218.1667
#
#Fit exponential smoothing model
#############
beta      <- seq(.0001, .5, by = .001)
RMSE      <- NA
for(i in seq_along(beta)) {
  fit     <- holt(product_data_agg_2017_watermelonKiran_ts[1:22], beta = beta[i], h = 4)
  RMSE[i] <- accuracy(fit, product_data_agg_2017_watermelonKiran_ts[23:26])[2,2]
}

# convert to a data frame and idenitify min alpha value
beta.fit <- data_frame(beta, RMSE)
beta.min <- filter(beta.fit, RMSE == min(RMSE))

# plot RMSE vs. alpha
library(ggplot2)
ggplot(beta.fit, aes(beta, RMSE)) +
  geom_line() +
  geom_point(data = beta.min, aes(beta, RMSE), size = 2, color = "blue")  

#We get optimum value of beta around 0.236. Building holts model around this.
watermelonKiran_holt <- holt(product_data_agg_2017_watermelonKiran_ts[1:22], beta = 0.236, h = 4)
MAPE(watermelonKiran_holt$mean, product_data_agg_2017_watermelonKiran_ts[23:26])
#MAPE with Holts model is 41.20898

#Building Holts winters model with Beta =0.027
watermelonKiran.smooth <- HoltWinters(product_data_agg_2017_watermelonKiran_ts[1:22], beta=0.236, gamma = FALSE)
plot(watermelonKiran.smooth$fitted)
watermelonKiran.smooth_pred <- forecast(maggi.smooth, h=4)
accuracy(watermelonKiran.smooth_pred,product_data_agg_2017_watermelonKiran_ts[23:26])
#MAPE from Holts Winters on test set is 1928.57

movAvgsimple       <- movavg(product_data_agg_2017_watermelonKiran_ts[1:24], 3, "s")
predmovAvgsimple   <- (product_data_agg_2017_watermelonKiran_ts[24] + product_data_agg_2017_watermelonKiran_ts[23] + product_data_agg_2017_watermelonKiran_ts[22]) / 3
MAPE(predmovAvgsimple, product_data_agg_2017_watermelonKiran_ts[25])
# MAPE is 2.729

# Predicting the previous data point
movAvgsimple       <- movavg(product_data_agg_2017_watermelonKiran_ts[1:23], 3, "s")
predmovAvgsimple   <- (product_data_agg_2017_watermelonKiran_ts[23] + product_data_agg_2017_watermelonKiran_ts[22] + product_data_agg_2017_watermelonKiran_ts[21]) / 3
MAPE(predmovAvgsimple, product_data_agg_2017_watermelonKiran_ts[24])
# MAPE is 1.83
#Best results for predicting weekly watermelon demand are from simple moving average

#############
