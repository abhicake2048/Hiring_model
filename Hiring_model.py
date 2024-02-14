import pandas as pd
import math
from sklearn.linear_model import LinearRegression
from word2number import w2n
 

d = pd.read_csv('hiring.csv')
x='zero'
a=[]
d.exp = d.exp.fillna(x)
for y in d.exp:
    a.append(w2n.word_to_num(y))
d.exp = a
print(d.exp)
m = math.floor(d.test_score.mean())
d.test_score = d.test_score.fillna(m)
reg = LinearRegression()
reg.fit(d[['exp','test_score','interview_score']],d.salary)
print(reg.predict([[12,10,10]]))