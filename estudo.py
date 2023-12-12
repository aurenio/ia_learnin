from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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

sns.scatterplot(x="horas", y="valor", data=url_info, hue="comprado")
plt.savefig("prova_1.png")

double = sns.FacetGrid(data=url_info, hue="comprado", col="comprado")
double.map(sns.scatterplot, "horas", "valor")
plt.savefig("prova_2_tentativa.png")