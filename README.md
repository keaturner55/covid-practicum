<center>

# **Cardiovascular Events Following Covid-19**
Keaton Turner<br>
Regis University<br>
MSDS 692: Practicum 1<br>
kturner006@regis.edu<br>
</center>


***


## **Project Overview**
The School of Pharmacy at Regis University partnered with the MSDS program to investigate cardiac events following a positive COVID diagnosis to confirm the hypothesis that SARS-CoV-2 can infect cells lining arteries and trigger inflammation and vascular hypertrophy. They were particularly interested in using social media data to identify individuals reporting a positive diagnosis. The next steps would then involve identifying users reporting issues associated with cardiac events and establishing a timeline between date of infection and symptoms of interest.


## **Methodology**
This project involved a great deal of text and natural language processing (NLP)--all done in Python. The steps taken to solve this problem were
1. Identify viable social media platform/data source for analysis
2. Web Scraping: Gather and clean as many social media posts as possible
3. Identify all users self-reporting symptoms of interest
4. Take the subset of users from step 3 and identify those reporting a positive date of infection
5. Create a timeline for each user between positive reporting dates and dates for symptoms of interest



## **Notebook Directory and Process Overview**
### [Facebook Post and Group Scraping](fb_scraper.ipynb)

The scraping step involved collecting post data from Facebook groups and storing the raw results in csv files. The API/utility only allowed for scraping of specific pages and groups, so the groups were hand-selected:
- COVID-19 Long Haulers Support
- COVID-19 The Long Haulers
- POST COVID SYMPTOMS

The scraper iterates through each post then recursively searches for comments and replies within the post object. Results were dumped into several csv files.
- Facebook scraping module: https://github.com/kevinzg/facebook-scraper

### [Text Cleaning and Preprocessing](fb_post_cleaning.ipynb)
This notebook was created for pre-processing and cleaning of the initial raw post data. A large portion of the effort involved extracting comments and replies from the post data, and creating separate data sets for them. The cleaned data was uploaded to an SQLite database (Facebook.db).

 ### [Identify Users Self-Reporting Symptoms of Interest](nlp_sentence_transformer_self_report.ipynb)
 This was the first real NLP step, in which users were identified as self-reporting symptoms of interest by iterating through sentences and comparing them to dummy sentences containing keywords for symptom categories and using a pre-trained BERT machine learning model. Results from this step were saved to a smaller database (Facebook_Self_Report.db).

- Sentence Transformer ML Module: https://www.sbert.net/
### [Identify Users Self-Reporting Positive Date of Infection](nlp_sentence_transformer_positive.ipynb)
This step was similar in method to identifying users with symptoms of interest: The same Sentence Transformer module was used with dummy sentences to find users reporting positive infection (e.g. "I tested positive"). But then extracting the date of infection was pretty much a manual step. Results from this step were appended to the Facebook_Self_Report.db.

### [Final Analysis and Timeline Creation](final_analysis.ipynb)
The final analysis code pulls from all of the primary data sources and does some basic analysis and visualization for all of the previous steps of the project. The last portion of this notebook then joins together the self-reporting symtpoms of interest and self-reporting positive date of infection datasets to create a final dataset of users for which a timeline can be established.

## Conclusion

Several assumptions and sources of error throughout this project could lead to skewed results. For example, in the self-reporting symptoms of interest code, the BERT model would at times match with sentences containing negation words (e.g. "I **haven't** experienced heart problems), so there could have easily been erronious data points in the final data set. Self-reporting of symptoms and positive diagnosis is also anecdotal at best. Cosine similarity scores are also not bullet-proof, it all depends on how accurate your ML model is for creating the sentence embedding vectors. Some input sentences matched with dummy sentences were flat out wrong. I tried removing many semi-manually, but I didn’t have time to go through and check the thousands of them.

Creating a timeline is also hard to automate unless you use timestamps only—which is unreliable unless you account for verb-tense in the NLP. In the end, creating custom ML models for separately identifying self-reporting symptoms and self-reporting positive cases would probably lead to more reliable results, BUT both of these tasks are huge undertakings by themselves. So I personnaly think that a great use of this type of study is to quickly gather lots of semi-relevant data and automate some key data extraction, but ultimately you would want to go through most of the data manually to check for errors befor using it in an official study.


