
README

################# CONTENT

This folder contains 12 files.
In particular, there are 3 files (train, val, test) per each of the 4 settings (left, right, starget, s3)

Files contain text in the first field and target quantifier in the second field.
Fields are tab-delimited

################# SETTINGS

# LEFT:
- in this setting, datapoints are made by the target sentence (containing <qnt>) and the preceding sentence
- max length: 100 tokens punctuation included

# RIGHT:
- in this setting, datapoints are made by the target sentence (containing <qnt>) and the following sentence
- max length: 100 tokens punctuation included

# STARGET:
- in this setting, datapoints are made by the target sentence (containing <qnt>) only
- max length: 50 tokens punctuation included

# S3:
- in this setting, datapoints are made by the target sentence (containing <qnt>), preceding sentence, and following sentence
- - max length: 150 tokens punctuation included

#################
