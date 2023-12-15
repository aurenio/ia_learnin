from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#exemplo de ia não adequada para o tipo de informação
url = "https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"
url_info = pd.read_csv(url)

rename_info = {
    "unfinished" : "nao_comprado",
    "expected_hours" : "horas",
    "price" : "valor"
}

url_info = url_info.rename(columns=rename_info)


troca = {
    0 : 1,
    1 : 0
}

url_info["comprado"] = url_info["nao_comprado"].map(troca)

info_x = url_info[["horas", "valor"]]
info_y = url_info[["comprado"]]
seed = 20
treino_x, teste_x, treino_y, teste_y = train_test_split(info_x, info_y, test_size=0.25, random_state=seed, stratify=info_y)

estudo = LinearSVC(random_state=seed)
estudo.fit(treino_x, treino_y)
chute_bot = estudo.predict(teste_x)
chute_comparacao = np.ones(540)

acerto_bot = accuracy_score(chute_bot, teste_y)
acerto_comparacao = accuracy_score(chute_comparacao, teste_y)

print(f"O bot acertou {acerto_bot} e tem que superar {acerto_comparacao}")

x_min = teste_x.horas.min()
x_max = teste_x.horas.max()
y_min = teste_x.valor.min()
y_max = teste_x.valor.max()

pixel = 100
eixo_x = np.arange(x_min, x_max, (x_max - x_min) / pixel)
eixo_y = np.arange(y_min, y_max, (y_max - y_min) / pixel)
xx, yy = np.meshgrid(eixo_x, eixo_y)
pontos = np.c_[xx.ravel(), yy.ravel()]

z = estudo.predict(pontos)
z = z.reshape(xx.shape)
print(z)

sns.scatterplot(y="valor", x="horas",  hue=teste_y.values.flatten(), data=teste_x)
plt.savefig("estudo_3")


plt.scatter(teste_x.horas, teste_x.valor, c=teste_y.values.flatten(), s=1)
plt.contourf(xx, yy, z, alpha=0.2)
plt.savefig("mygod.png")

sns.scatterplot(x="horas", y="valor", data=url_info, hue="comprado")
plt.savefig("prova_1.png")

double = sns.FacetGrid(data=url_info, hue="comprado", col="comprado")
double.map(sns.scatterplot, "horas", "valor")
plt.savefig("prova_2_tentativa.png")





teste