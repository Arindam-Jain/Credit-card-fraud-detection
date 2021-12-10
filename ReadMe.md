# Fake News Detection
========
It is written in Python

Get from PyPI but only python3.0 or above
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This is the simplest (one-command) install method 

.. code:: shell

   pip3 install -r requirements.txt 
   The above command will install all the required libraries required to execute this code

# Steps 
~~~~~~~~~~~~

1. The first step that we perform is data visualization. We check for inconsistent data like duplicates, missing values, and incorrect urls.

2. From the urls given in the dataset, we extract the text contained in it by using a web scraping library named “Beautiful Soup”.

3. A few websites contained text in languages other than English(German, Chinese, Portuguese,..). So we translated such data into English.

4. The text extracted from the urls is helpful for generating better features in order to train the model effectively.

5. We then eliminated the stop words like “a”, “the”, “is”, “are”  from the claim and extracted text. This was done because the stop words appear multiple times in the data and carry very little information or are of no value. 

6. We further cleaned the noise in the data like commas, dots, ids, deleting the suffixes by stemming terms using NLP NLTK libraries.

7. Generated the website name from the entire url in order to analyse which websites provide real and fake information.


# Folder Structure

.
├── Code                     
│   ├── bert_tpu.py          # bert model run with tpu in python
│   ├── CNN_gpu.py           # CNN model run with gpu in python
│   ├── ML_models.py         # Random Forest,Decision Tree, Logistic model in python  
│      
├── Dataset                     
│   ├── test_translated.csv          # cleaned train files after scraping data from web-link
│   ├── train_translated.csv         # cleaned test files after scraping data from web-link
│   
├── smmfall21asu                     
│   ├── train.csv                   #  train files after scraping data from web-link
│   ├── test.csv                    #  test files after scraping data from web-link
│   ├── sample_submission.csv       #  submission files after scraping data from web-link
│  
├── ReadMe.md
│ 
├── requirement.txt


