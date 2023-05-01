#%%
import numpy as np
import seaborn as sns
import pandas as pd
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
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
