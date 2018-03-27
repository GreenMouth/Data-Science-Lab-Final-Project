# Data-Science-Lab-Final-Project
---
## Wutz dis repo
put stuff here

## Feel free to set up on your machine however you want
But note I added a .gitignore if you want to use it, it ignores virtual environment files.

(If you want) You can make a virtual environment by

```
# Make a virtual environment
virtualenv -p python3 Data-Science-Lab-Final-Project

# Download the repo
git clone git@github.com:jsigee87/Data-Science-Lab-Final-Project.git

# Rename it if you want
mv Data-Science-Lab-Final-Project <whateveryouwanttonameit>

cd <newdirname>

# activates your virtual environment
. bin/activate

# reads a text file called requirements and installs all the 
# libraries listed in it (the ones we need)
pip install -r requirements

# now you have a self contained python environment. try
jupyter notebook

# to leave the virtual environment
deactivate
```
