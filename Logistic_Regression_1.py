#%%
import numpy as np
import seaborn as sns
import pandas as pd
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#%%
with open('C:/Users/User/OneDrive/桌面/Github/math/iris_2.csv',newline='') as csvfile:
    irisData=csv.reader(csvfile)
    irisDF=pd.DataFrame(irisData)
csvfile.close()
irisDF.columns=['sepal length', 'sepal width',
                'petal length','petal width','class']

print(irisDF)
#%%    
g=sns.scatterplot(x='petal length', y='petal width', hue='class', data=irisDF)
g.figure.axes[0].invert_yaxis()
plt.show()
# %%
x=irisDF[['petal length','petal width']]
y=irisDF['class']
x
# %%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x_arr=np.linspace(-10,10,100)
y_arr=np.array(list(map(sigmoid,x_arr)))
# %%

fig,ax=plt.subplots(figsize=(10,6))
ax.plot(x_arr,y_arr)
ax.set_xticks([-10,0,10],['$-\infty$','0','$\infty$'])
ax.set_yticks([0,0.5,1],['0','0.5','1'])
ax.set_xlabel('$x$')
ax.set_ylabel('$Sigmoid(x)$')
ax.set_title('Sigmoid Function')
ax.grid()
plt.show()
# %%
le=LabelEncoder()
y_encoded=le.fit_transform(y)
y_encoded
# %%
X_train, X_test, Y_train, Y_test = train_test_split(x, y_encoded, test_size=0.2)
#%%
logR=LogisticRegression(random_state=0)
logR.get_params()
#%%
logR.fit(X_train,Y_train)
logR.score(X_train,Y_train)
#%%
logR.coef_
#%%
logR.intercept_
#%%
def fun(x1,x2):
    return 2.47337199*x1+1.03289823*x2-7.44987506
def sigmoid_new(x1,x2):
    return 1/(1+np.exp(-(fun(x1,x2))))


# %%
X_train.head(6)
#%%
Y_train[0:6]
#%%
outFunYList=[]
outSigmoidYList=[]
test_list=[[4.1,1.3],
           [3.9,1.1],
           [3.7,1.0]]
print('線性函數:f(x)結果','接近1表示Sentosa,接近0表示Versicolor')
for i in test_list:
    outFunYList.append(fun(i[0],i[1]))
    outSigmoidYList.append(sigmoid_new(i[0],i[1]))
    print(fun(*i),"\t",sigmoid_new(*i))

# %%
