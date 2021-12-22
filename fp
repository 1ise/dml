                 FP_growth program

#to install mlxtend use 'pip install mlxtend'
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
 ['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
 ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
 ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
 ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
print(df.head())
# 60% minimum support
tb_df = fpgrowth(df, min_support=0.6, use_colnames=True)
print(tb_df)





Apple Corn Eggs Ice cream ... Nutmeg Onion Unicorn Yogurt
0 False False True False ... True True False True
1 False False True False ... True True False True
2 True False True False ... False False False False
3 False True False False ... False False True True
4 False True True True ... False True False False
[5 rows x 10 columns]
 
 support itemsets
0 1.0 (Kidney Beans)
1 0.8 (Milk)
2 0.8 (Eggs)
3 0.6 (Yogurt)
4 0.6 (Onion)
5 0.8 (Milk, Kidney Beans)
6 0.8 (Eggs, Kidney Beans)
7 0.6 (Eggs, Milk)
8 0.6 (Eggs, Milk, Kidney Beans)
9 0.6 (Yogurt, Milk)
10 0.6 (Yogurt, Kidney Beans)
11 0.6 (Yogurt, Milk, Kidney Beans)
12 0.6 (Eggs, Onion)
13 0.6 (Onion, Kidney Beans)
14 0.6 (Eggs, Onion, Kidney Beans)
