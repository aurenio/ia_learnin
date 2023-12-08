import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

url = pd.read_csv("https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv")

dates = url[["expected_hours", "price"]]

buy_finish = url[["unfinished"]]

SEED = 20
treino_x, teste_x, treino_y, teste_y = train_test_split(dates, buy_finish, random_state=SEED, test_size=0.25,stratify=buy_finish)

model = LinearSVC()

model.fit(treino_x, treino_y)

result = model.predict(teste_x)
result_score = accuracy_score(result, teste_y)

print(result_score)