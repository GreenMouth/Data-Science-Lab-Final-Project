# What Are These Files?

All of the source code for the project is in this directory. Here is a short description of the files.

<ul>

<li>

`clean.sh` chains together all the individual data preparation scripts we wrote. If you run this script it will do all the data cleaning and write a file called merged-data.csv to the data/kaggle-data directory.

</li>

<li>

`data-cleaning` has all the scripts we used for our initial cleaning of our data and initial generation of new features like sentiment score and TFIDF vectors for each transcript 

</li>
<li>

`lstm_generation` contains all the code and results of us attempting to train lstm networks to generate their own TED talks

</li>
<li>

`lstm_proj` contains all the code and results of us training lstm networks to classify the gender of a particular speaker through speech patterns

</li>
<li>

`Data-Exploration.ipynb` contains a full exploration of the data, including relevant figures and graphs. 

</li>
<li>

`graphs` has a copy of some prepared graphs that were used in our report.

</li>

</ul>
