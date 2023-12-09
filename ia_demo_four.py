from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#tentativa de previsão de projeto finalizado de acordo com um historico em csv 


url = "https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"

url_info = pd.read_csv(url)


rename_info = {
    "unfinished" : "nao_terminados",
    "expected_hours" : "espectativa_horas",
    "price" : "valor"
}

url_info = url_info.rename(columns=rename_info)

troca = {
    0 : 1,
    1 : 0
}


url_info["finalizados"] = url_info.nao_terminados.map(troca)


sns.scatterplot(x = "espectativa_horas", y = "valor", data = url_info, hue = "finalizados")

plt.savefig("grafico.png")


zero_one = sns.FacetGrid(data=url_info,hue="finalizados", col="finalizados")
zero_one.map(sns.scatterplot, "espectativa_horas", "valor" )
plt.savefig("comparacao")


sucess = url_info[["finalizados"]]

unsucess = url_info[["nao_terminados"]]

data = url_info[["espectativa_horas", "valor"]]

seed = 20

treino_x, teste_x, treino_y, teste_y = train_test_split(data, sucess, test_size=0.25, random_state=seed, stratify=sucess)

model = LinearSVC()

model.fit(treino_x, treino_y)

analis = model.predict(teste_x)

print(accuracy_score(analis, teste_y))

previ_base = np.ones(540)

chute = accuracy_score(analis, previ_base)

print(f"meta a superar {chute}")