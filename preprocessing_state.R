library(stringr)
library(tidyverse)
#library(readr)
#wd <- "⁩⁩~/Users/tkousser/Dropbox/SummerofTweets/master_classifier"
# setwd('~/Downloads/TweetsClassification')
#setwd(wd)
data <- read.csv('training_data_stateprez.csv')
# dput(colnames(data))
attach(data)

names_to_remove = c('america next','believeagaingop',
                    'berniesanders','bobbyjindal',
                    'carlyfiorina','chrischristie',
                    'correctrecord','draftrunbenrun',
                    'genfwdpac','govmikehuckabee',
                    'governorpataki','governorperry',
                    'hillaryclinton','jebbush',
                    'jimwebbusa','johnkasich',
                    'keeppromise1','lessing2016',
                    'lincolnchafee','lindseygrahamsc',
                    'martinomalley','millenniarise',
                    'newday4america','oppandfreedom',
                    'ourrevival','pagpac','randpaul',
                    'realbencarson','rebuildingamnow',
                    'ricksantorum','scottwalker',
                    'the purple pac','unintimidated16',
                    'workingagainpac','america leads',
                    'americasliberty','carlyforamerica',
                    'cspac','feelthebernorg','gov gilmore',
                    'greatamericapac','marcorubio','mkdclsn',
                    'prioritiesUSA','progressivekick',
                    'r2rusa','realdonaldtrump',
                    'securestrength','tedcruz','tedcruz45')


#lowercase
processed_text <- tolower(text)
# # Get rid of URLs
processed_text <- gsub(" ?(f|ht)(tp)(s?)(://)(.*)[.|/](.*)", "", processed_text)
# #get rid of unnecessary spaces
processed_text <- str_replace_all(processed_text," "," ")
# # remove punctuations
processed_text <- gsub('[[:punct:] ]+',' ',processed_text)
#drop screennames
for (name in  names_to_remove){
  processed_text <- str_replace_all(processed_text,name,"")
}

data$text <- processed_text

write_csv(data,"training_data_stateprez_processed.csv")

# # # Take out retweet header
# # clean_tweet <- str_replace(clean_tweet,"rt @[a-z,A-Z]*: ","")
# # # Get rid of hashtags
# # clean_tweet <- str_replace_all(clean_tweet,"#[a-z,A-Z]*","")
# # Get rid of references to other screennames
# # processed_text <- str_replace_all(processed_text,"@[a-z,A-Z]*","")
# # write_csv(x=data,path="ProcessedDataDropScreenNames.csv")

# ###########################################
# setwd('~/tweetsclassification')
# data <- read.csv('ProcessedDataLowNoLinkNoPunc.csv')
# # dput(colnames(data))
# attach(data)

# processed_text <- text
# for (name in  names_to_remove){
#   processed_text <- str_replace_all(processed_text,name,"")
# }

# data$text <- processed_text

# write_csv(x=data,path="ProcessedDataLowNoLinkNoPuncNoNames.csv")
############################
data <- read_csv('State_Leg_Tweets_corpus.csv') %>%
  dplyr::select(-c(1:2,6:8)) %>%
  rename("id"=1,
         "created_at"=2,
         "text"=3,
         "likes"=4,
         "rts"=5,
         "user"=6)
# dput(colnames(data))

# lowercase
processed_text <- tolower(data$text)
# Get rid of URLs
processed_text <- gsub(" ?(f|ht)(tp)(s?)(://)(.*)[.|/](.*)", "", processed_text)
#get rid of unnecessary spaces
processed_text <- str_replace_all(processed_text," "," ")
# remove punctuations
processed_text <- gsub('[[:punct:] ]+',' ',processed_text)
#drop screennames
# for (name in  names_to_remove){
#   processed_text <- str_replace_all(processed_text,name,"")
# }

data$text <- processed_text

write_csv(data,"ProcessedFullCorpus_state.csv")
