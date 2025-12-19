# %%

import pandas as pd

df = pd.read_excel("data/dados_cerveja_nota.xlsx")

df.head()
# %%

from sklearn import linear_model
from sklearn import tree

X = df[["cerveja"]] #Isso aqui é uma matriz (Dataframe)
y = df["nota"] # Isso é um vetor (Serie)

# Isso é o aprendizado de maquina
reg = linear_model.LinearRegression()
reg.fit(X,y)

# %%

a, b = reg.intercept_, reg.coef_[0]

predict_reg = reg.predict(X.drop_duplicates())

arvore_full = tree.DecisionTreeRegressor(random_state=42)

arvore_full.fit(X, y)
predict_arvore_full = arvore_full.predict(X.drop_duplicates())

arvore_d2 = tree.DecisionTreeRegressor(random_state=42,
                                       max_depth=2)

arvore_d2.fit(X, y)
predict_arvore_d2 = arvore_d2.predict(X.drop_duplicates())

# %% 
predict = reg.predict(X.drop_duplicates())

# %%

import matplotlib.pyplot as plt

plt.plot(X["cerveja"], y, "o")
plt.grid(True)
plt.title("Relação Cerveja vs Nota")
plt.xlabel("Cerveja")
plt.ylabel("Nota")

plt.plot(X.drop_duplicates()["cerveja"], predict_reg)

plt.plot(X.drop_duplicates()["cerveja"], predict_arvore_full, color="green")

plt.plot(X.drop_duplicates()["cerveja"], predict_arvore_d2,color="magenta")

plt.legend(["Observado", 
            f"y = {a:.3f} + {b:.3f}  x",
            "Arvore Full",
            "Arvore d2"
            ])
# %%

plt.figure(dpi=200)
tree.plot_tree(arvore_full,
               feature_names=["cerveja"],
               filled=True)

# %%
arvore_full.predict([[10]])

# %%
