### agatston score distribution

+ dataset: Totalsegmentator




+ for data selection criteria: see `agg.py`.


raw (makes no sense):

<img src="./agatston_score_raw.png">

filtered:

```
df = df[(df.median_hu<60)&(df.mask_volume>100000)] # ??
```


<img src="./agatston_score.png">