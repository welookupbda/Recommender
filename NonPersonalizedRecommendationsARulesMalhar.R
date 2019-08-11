# https://rpubs.com/sbushmanov/180410

library(arules)
library(arulesViz)

GroceriesFull <- read.transactions('/Users/svattiku/Google Drive/BDA/ProjectBDA/Programs/Latest/Full_transactions_product_descriptions_MM-INDORE-MALHAR MEGA MALL.csv', sep='\t')
GroceriesFiltered <- read.transactions('/Users/svattiku/Google Drive/BDA/ProjectBDA/Programs/Latest/Filtered_transactions_product_descriptions_MM-INDORE-MALHAR MEGA MALL.csv', sep='\t')

GroceriesFull90days <- read.transactions('/Users/svattiku/Google Drive/BDA/ProjectBDA/Programs/Latest/Full90days_transactions_product_descriptions_MM-INDORE-MALHAR MEGA MALL.csv', sep='\t')
GroceriesFiltered90days <- read.transactions('/Users/svattiku/Google Drive/BDA/ProjectBDA/Programs/Latest/Filtered90days_transactions_product_descriptions_MM-INDORE-MALHAR MEGA MALL.csv', sep='\t')

GroceriesFull120days <- read.transactions('/Users/svattiku/Google Drive/BDA/ProjectBDA/Programs/Latest/Full120days_transactions_product_descriptions_MM-INDORE-MALHAR MEGA MALL.csv', sep='\t')
GroceriesFiltered120days <- read.transactions('/Users/svattiku/Google Drive/BDA/ProjectBDA/Programs/Latest/Filtered120days_transactions_product_descriptions_MM-INDORE-MALHAR MEGA MALL.csv', sep='\t')

summary(GroceriesFull)
summary(GroceriesFiltered)
summary(GroceriesFull90days)
summary(GroceriesFiltered90days)
summary(GroceriesFull120days)
summary(GroceriesFiltered120days)

getFrequentSets15 <- function(dataset, supportValue, minLen, maxLen) {
                         itemsets <- apriori(dataset,
                         parameter = list(support=.001,
                                       minlen=minLen,
                                       maxlen=maxLen,
                                       target='frequent' # to mine for itemsets
                         ))
  
                         summary(itemsets)
  
                         inspect(sort(itemsets, by='support', decreasing = T)[1:15])
                     }

getFrequentSets15(GroceriesFull, 0.001,3,3)
getFrequentSets15(GroceriesFiltered, 0.001,3,3)
getFrequentSets15(GroceriesFiltered120days, 0.001,3,3)

getRules <- function(dataset, supportValue, confidence, minLen, maxLen, measure) {
               rules <- apriori(dataset,
                   parameter = list(support=supportValue,
                                    confidence=confidence,
                                    minlen=minLen,
                                    maxlen=maxLen,
                                    target='rules' # to mine for rules
                   ))
               summary(rules)
               inspect(sort(rules, by=measure, decreasing = T)[1:15])
            }

getRules(GroceriesFull, 0.001, 0.5, 1,5, 'lift')
getRules(GroceriesFiltered, 0.001, 0.5, 1,5, 'lift')
getRules(GroceriesFiltered120days, 0.001, 0.5, 1,5, 'lift')
getRules(GroceriesFull120days, 0.001, 0.5, 1,5, 'lift')
