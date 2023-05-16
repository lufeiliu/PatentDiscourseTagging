# PatentDiscourseTagging

TEST RESULTS

8 patents, 2599 sentences

Fine tune BerforPatent
1. cls + linear 

| folds | precision | recall | f-score |
|-------|-----------|--------|---------|
| 1     | 0.71      | 0.63   | 0.65    |
| 2     | 0.73      | 0.66   | 0.67    |
| 3     | 0.72      | 0.63   | 0.64    |
| 4     | 0.72      | 0.61   | 0.61    |

2. cls of last 4 hidden layers + linear 

| folds | precision | recall | f-score |
|-------|-----------|--------|---------|
| 1     | 0.73      | 0.65   | 0.67    |
| 2     | 0.73      | 0.67   | 0.67    |
| 3     | 0.74      | 0.64   | 0.66    |
| 4     | 0.69      | 0.63   | 0.63    |

3. bert + crf (1 sequence = 3 sentences, using <sep> token)

| folds | precision | recall | f-score |
|-------|-----------|--------|---------|
| 1     | 0.66      | 0.6   | 0.61    |
| 2     | 0.73      | 0.66   | 0.68    |
| 3     | 0.74      | 0.66   | 0.68    |
| 4     | 0.68      | 0.64   | 0.64    |
  
