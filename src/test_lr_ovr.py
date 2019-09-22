from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
X=[]
ids=[]
with open('../data/gw.emb')as f:
    f.readline()
    for line in f:
        line=line.strip().split(' ')
        id=int(line[0])
        vec=list(map(float,line[1:]))
        X.append(vec)
        ids.append(id)
yd={}
with open('../data/usa-airports.group')as f:
    for line in f:
        line=line.strip().split(' ')
        id,l=list(map(int,line))
        yd[id]=l
y=[yd[_] for _ in ids]

train_ratio=0.8
train_num=int(len(X)*train_ratio)
test_num=len(X)-train_num
train_X=X[:train_num]
train_y=y[:train_num]
test_X=X[train_num:]
test_y=y[train_num:]
clf = LogisticRegression(random_state=0, max_iter=100000,solver='saga',penalty='l2')
clf=OneVsRestClassifier(clf).fit(train_X, train_y)
pred=clf.predict(test_X)
print(train_X)
print(train_y)
print(test_X)
print(test_y)
print(pred)
print(accuracy_score(test_y,pred))
#print(clf.coef_)
#print(clf.intercept_)