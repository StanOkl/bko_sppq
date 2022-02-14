##This is how you add comments in Python
##twitter_data_compiler is a python script that runs 4 different scripts, 3 python and 1 R.



##The first thing that happens is we import packages
##This is the same as library() in R
##subprocess lets us run command line via Python

import subprocess
import zipfile
import urllib.request
 
#url = 'https://www.dropbox.com/sh/vnu9linsuv5ouxe/AABSs1KHdw3fjPnY7TAIoHFPa?dl=1' 
#urllib.request.urlretrieve(url, 'new_tweets.zip') 
 
#with zipfile.ZipFile("new_tweets.zip","r") as zip_ref:
    #zip_ref.extractall()
 
#exec(open("convert_x.py").read())
 
subprocess.call (["Rscript", "--vanilla", "preprocessing_state.R"])
 
exec(open("TFIDF_ClassificationsState.py").read())
 
#exec(open("polling_follower_merge.py").read())
