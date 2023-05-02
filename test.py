import pandas as pd

# 创建一个 DataFrame 对象
df = pd.DataFrame({'A': ['1', '2', '3'], 'B': ['4', '5', '6']})

# 使用 values.tolist() 方法将 DataFrame 转换为嵌套列表
list_data = df.values.tolist()
print(df)
print(list_data)
list_data=[list(map(int,i)) for i in list_data]
print(list_data)
