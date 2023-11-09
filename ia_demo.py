from sklearn.svm import LinearSVC

#features (1 = sim, 0 = não)
# long hair
# short leg
# barks

pig1 = [0, 1, 0]
pig2 = [0, 1, 1]
pig3 = [1, 1, 0]

dog1 = [0, 1, 1]
dog2 = [1, 0, 1]
dog3 = [1, 1, 1]

data = [pig1, pig2, pig3, dog1, dog2, dog3]

category = [1, 1, 1, 0, 0, 0]