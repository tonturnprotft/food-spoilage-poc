import pandas as pd, datetime as dt, itertools, pathlib

rows=[]
mapping={1:1,2:2,3:3,4:4,5:5,6:4,7:5,8:5,9:5,10:5,11:5}
start=dt.date(2024,1,19)
for offset,score in mapping.items():
    day=(start+dt.timedelta(days=offset-1)).isoformat()
    for sess in (1,2):
        rows.append({"day":day,"session":sess,"species":"tuna","score":score})
df=pd.DataFrame(rows)
dst=pathlib.Path("data/raw/dafif/labels"); dst.mkdir(parents=True,exist_ok=True)
df.to_csv(dst/"organoleptic_scores.csv",index=False)
print("✓ stub label file written")