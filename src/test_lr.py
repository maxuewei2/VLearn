from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
X=[]
y=[]
with open('../data/breast-cancer-wisconsin.data1')as f:
    for line in f:
        line=line.strip().split(',')
        line=list(map(int,line))
        X.append(line[1:10])
        y.append(line[-1])
train_ratio=0.8
train_num=int(len(X)*train_ratio)
test_num=len(X)-train_num
train_X=X[:train_num]
train_y=y[:train_num]
test_X=X[train_num:]
test_y=y[train_num:]
clf = LogisticRegression(random_state=0, solver='sag').fit(train_X, train_y)
pred=clf.predict(test_X)
print(train_X)
print(train_y)
print(test_X)
print(test_y)
print(pred)
print(accuracy_score(test_y,pred))
print(clf.coef_)
print(clf.intercept_)