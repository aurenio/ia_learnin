from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
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

print(date)

print(date_bought)