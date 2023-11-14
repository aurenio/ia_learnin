from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
#features (1 = yes, 0 = no)
# long hair
# short leg
# barks
#LinearSVC = estimador
#1 = pig, 0 = dog

pig1 = [0, 1, 0]
pig2 = [0, 1, 1]
pig3 = [1, 1, 0]

dog1 = [0, 1, 1]
dog2 = [1, 0, 1]
dog3 = [1, 1, 1]

data = [pig1, pig2, pig3, dog1, dog2, dog3]

category = [1, 1, 1, 0, 0, 0]

model = LinearSVC()
model.fit(data, category)

mystery = [1,1,1]

resp = model.predict([mystery])

print(resp)
print("the result is 0")
#com mystery setados em 1, 1, 1. estamos falando para o codigo que o animal misterioso tem perna curta, cabelo longo e late. Claramente um cachorro pelas informações 
#dispostas, a resposta correta tem que vim com um 0

mystery_1 = [1, 1, 0]
mystery_2 = [0, 1, 1]

test_all = [mystery, mystery_1, mystery_2]

resp_preview = model.predict(test_all)
print(resp_preview)
print("the result is 0, 1, 0")

#agora vamos dizer que aparecer um imprevisto, esse imprevisto é o último resultado e que esperavamos 0, 1, 1, vamos mostrar o quanto o modelo conseguiu acertar
# e agora vamos ver o quanto ele preveu
correct = [0, 1, 1] 


# result_sum = result.sum()

# print(result_sum)

# print("true = 1, false = 0. result = 2")

# all = len(result)

# hit_rate = (result_sum/all) * 100

# print(hit_rate)

hit_rate = accuracy_score(correct, resp_preview)

print(hit_rate * 100)