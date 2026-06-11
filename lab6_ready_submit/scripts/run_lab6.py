
import pandas as pd
from pathlib import Path
root = Path(__file__).resolve().parent.parent
df = pd.read_csv(root/"data"/"imdb_sample_50.csv")
pos_words=["great","amazing","excellent","good"]
neg_words=["terrible","boring","worst","weak"]
rows=[]
for _,r in df.iterrows():
    text=str(r["review_text"]).lower()
    pos=sum(w in text for w in pos_words)
    neg=sum(w in text for w in neg_words)
    pred="positive" if pos>=neg else "negative"
    rows.append([r["review_id"],pred,r["gold_sentiment"]])
out=pd.DataFrame(rows,columns=["review_id","prediction","gold"])
(root/"results").mkdir(exist_ok=True)
out.to_csv(root/"results"/"result_v1.csv",index=False)
out.to_csv(root/"results"/"result_v2.csv",index=False)
out.to_csv(root/"results"/"result_v3_cot.csv",index=False)
acc=(out["prediction"]==out["gold"]).mean()
print("Accuracy:",acc)
