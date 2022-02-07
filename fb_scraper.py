from datetime import datetime
from facebook_scraper import get_posts, set_cookies, exceptions
import time
import pandas as pd
import json
import argparse

# %%
set_cookies("cookies.json")

# %%
results = []
start_url = None

def handle_pagination_url(url):
    global start_url
    start_url = url


# %%
results = []
start_url = None

def handle_pagination_url(url):
    global start_url
    start_url = url
i = 0
while True:
    try :
        for post in get_posts(group = "373920943948661", page_limit = 1,start_url = start_url,
        request_url_callback = handle_pagination_url):
            if (i%10)==0:
                print(i)
            results.append(post)
        print("Finished")
        break
    except exceptions.TemporarilyBanned:
        print("TEMPORARY BAN... sleeping for 30 min")
        all_posts = pd.DataFrame()
        for each in results:
            all_posts = all_posts.append(each, ignore_index = True)
        if len(all_posts)>0:
            all_posts.to_csv("fb_scraper_{}.csv".format(i))
        time.sleep(1800)


# %%
all_posts = pd.DataFrame()
for each in results:
    all_posts = all_posts.append(each, ignore_index = True)
    


# %%
all_posts.to_csv("fb_srcaper1.csv")

# %%
len(all_posts)

# %%
all_posts.head()

# %%



