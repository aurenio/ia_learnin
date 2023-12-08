from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd



#nesse novo arquivo utilizaremos a ia para analisar a lista de compras de um determinado site fazendo indicação de acordo de como as pessoas acessam o site
#home = acessar a pagina inicial,how_it_works = como o site funciona, contato = foram atras da parte do site para entrar em contato, bought = se comprou ou não algum produto
#1 significa que teve uma interação e 0 que não teve
#https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv
#nesse momento não estamos preocupados com a ordem de como ocorreu o acesso

url = "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"

url_csv = pd.read_csv(url)
date = url_csv[["home", "how_it_works", "contact"]]
date_bought = url_csv[["bought"]]

#para controle vamos separar uma quantidade de elementos, verificamos quantos elementos temos
print(url_csv.shape)

#vamos pegar 75 elementos para treinamento e o resto para teste

#nesse novo formato, estou definindo uma semente, que vai padronizar as informações aleatorias separadas que o comando train_test_split vai fazer para termos
#um controle de testes, assim vai melhorar a eficiência e otimizar o codigo, test_size é responsável pela quantidade de parâmetos para teste e para não deixar
#o codigo com vies errado, vamos estratificar de acordo com o resultado, assim teremos uma mesma proporção de quem compra e quem não compra

SEED = 20

treino_x, teste_x, treino_y, teste_y = train_test_split(date, date_bought, test_size=0.25, random_state=SEED, stratify=date_bought)


model = LinearSVC()
model.fit(treino_x,treino_y)
acurracy = model.predict(teste_x)
score = accuracy_score(teste_y, acurracy)

print(score)