### TFVectorRecommender.py
This script generates a matrix of TFIDF vectors of our transcripts then uses cosine similiarity to generate a column that appends onto our metadata dataframe data frame that list the top n most similar other TED talks.

### analyzeSentiment.py
This script uses TextBlob to anaylze the sentiment and subjectivity score of the scripts and appends that column to our metadata dataframe. The sentiment score is on a scale of -1 to 1 with those meaning very negative to very positive respectively.

### analyzeTFID.py
This script generates it's own dataframe of TFIDF vectors that it stores in a seperate csv from the main dataframe as they don't fit into a dataframe column very well

### cleanText.py
A lot of junk punctuation or symbols need to be cleaned from our data so they don't confuse our models. Clean text has a list of symbols and junk to look out for and swaps them out with new lines instead

### finalStrip.py
Final strip takes one last pass through our metadata dataframe to find and bad or NaN or NA values and drops the row from the dataset

### joinData.py
This merges our TED meta data dataframe with each of the transcripts from our transcript dataframe
