#Load packages
library(data.table)
library(tidyverse)
library(readxl)
library(irr)
library(reticulate)
library(haven)
##remove scientific notation
options(scipen=999)
options(dplyr.width=Inf)

setwd("bko_sppq_rep")
###Table 1 
####Measures of Intercoder Reliability: Humans Agreeing with Humans

###get human coded tweets
joined.2coders.data <- read_csv('Training_Tweets_8-9.csv')

###
df.tweets <- data.frame(joined.2coders.data[!is.na(joined.2coders.data$coder1.Sentiment....1.negative..0.neutral..1.positive.) &
                                              !is.na(joined.2coders.data$coder2.Sentiment....1.negative..0.neutral..1.positive.),])

df.tweets[is.na(df.tweets)] <- 0

df.tweets <- df.tweets[df.tweets$src !="0",]

###simple matches for topic matching

var.names <- names(df.tweets[c(5:21)])

##IRR loop

counter = 0
inter.rater.reliability <- NULL
inter.rater.reliability <- data.frame(variable=as.character(0), pct=as.numeric(0), 
                                      cohens.kappa=as.numeric(0),stringsAsFactors = F)

for (i in var.names){
  matches <- df.tweets[,5+counter]==df.tweets[,23+counter]
  y <- length(matches[matches==TRUE])/length(matches)
  inter.rater.reliability[counter+1,1] <- as.character(gsub("coder1.","",i))
  inter.rater.reliability[counter+1,2] <- y
  counter = counter +1
}

##cohen's kappa

counter = 0
for (i in var.names){
  x <- kappa2(df.tweets[,c(5+counter,23+counter)], "unweighted")
  inter.rater.reliability[counter+1,3] <- x$value
  counter = counter +1
}

###rename variables, reorder rows and output table

row_order <- c("Is the Tweet a Factual Claim or an Opinion?","Ideology (Liberal, Neutral, or Conservative)",
               "Sentiment (Negative, Neutral, or Positive)","Is the Tweet Political or Personal?",
               "Topic: Immigration","Topic: Macroeconomics",
               "Topic: Defense","Topic: Law and Crime","Topic: Civil Rights","Topic: Environment",
               "Topic: Education","Topic: Health","Topic: Government Operations",
               "Topic: No Policy Content","Asks for a Donation?","Asks to Watch, Share, Or Follow?",
               "Asks for Miscellaneous Action?")

inter.rater.reliability %>% 
  mutate(Name=c("Sentiment (Negative, Neutral, or Positive)","Is the Tweet Political or Personal?",
                "Ideology (Liberal, Neutral, or Conservative)","Topic: Immigration","Topic: Macroeconomics",
                "Topic: Defense","Topic: Law and Crime","Topic: Civil Rights","Topic: Environment",
                "Topic: Education","Topic: Health","Topic: Government Operations",
                "Topic: No Policy Content","Asks for a Donation?","Asks to Watch, Share, Or Follow?",
                "Asks for Miscellaneous Action?","Is the Tweet a Factual Claim or an Opinion?"),
         across(where(is.numeric), round,2)) %>%
  rename("Agreement Rate"=pct,
         "Cohen's Kappa"=cohens.kappa) %>%
  select(-variable) %>%
  relocate(Name) %>%
  slice(match(row_order, Name)) 

####Table 3
###Measures of Classification Accuracy: Computers Replicating Humans

##This runs a Python file that calls "preprocessing_state.R" which cleans the collected twitter data
##and outputs a csv that is read by "TFIDF_ClassificiationsState.py" which runs the ML classifications
##of the tweets based on our human-coded sample. The classification accuracy measures are outputted in a csv 
##file "accuracy_output.csv"

py_run_file("twitter_data_compiler_state.py")

####Table 4
###Does Gender Affect Twitter Activity?

data <- read_dta("GenderTwitterSPPQ.dta")









