dataset = pd.read_csv('python/data.csv')
dataset = dataset.drop(['id','Unnamed: 32'],axis=1)
dataset['diagnosis'] = [1 if x == 'M' else 0 for x in dataset['diagnosis']]
dataset_positive = dataset[dataset['diagnosis'] == 1]
dataset_negative = dataset[dataset['diagnosis'] == 0]

data_analysis(dataset)
data = dataset.corr()
print("features having correlation greater than 0.75 with target are")
print(data[data['diagnosis'] > 0.75].index)

x = dataset.copy()
y = x['diagnosis']
x = x.drop(['diagnosis'],axis=1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 1234)
sm = SMOTE(random_state = 1234)
X_sm,y_sm = sm.fit_resample(x_train,y_train)
print(X_sm.corr().to_string())
cols=['radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst','concave points_se']
X_sm = X_sm.drop(cols,axis=1)
x_test=x_test.drop(cols,axis=1)
cols1=['perimeter_mean','perimeter_se','area_mean','area_se','concave points_mean','compactness_mean','symmetry_mean','texture_se']
X_sm = X_sm.drop(cols1,axis=1)
x_test=x_test.drop(cols1,axis=1)
print("final columns")
print(X_sm.columns)

sc = StandardScaler()
X_sm = sc.fit_transform(X_sm)
x_test = sc.transform(x_test)

classifier_algorithms = [[RandomForestClassifier(),'Random Forest'],[KNeighborsClassifier(),'KNN'],[SVC(),'SVC'],[GaussianNB(),'naivebaiyes'],[DecisionTreeClassifier(),'Decision trees'],[LogisticRegression(),'logistic regresion'],[AdaBoostClassifier(),'AdaBoostClassifier']]
model1 = RandomForestClassifier()
model2 = KNeighborsClassifier()
model3 = SVC()
model4 = GaussianNB()
model5 = DecisionTreeClassifier()
model6 = LogisticRegression()
model7 = AdaBoostClassifier()
final_model = VotingClassifier(

    estimators = [('RFC',model1),('KNC',model2),('SVC',model3),('GNB',model4),('DTC',model5),('LR',model6),('ABC',model7)] , voting = 'hard'
)
final_model.fit(X_sm,y_sm)
y_prediction = final_model.predict(x_test)
print("score of majority voting is")
print(accuracy_score(y_test,y_prediction)*100)
print(precision_score(y_test,y_prediction)*100)
print("f1_Score",f1_score(y_test,y_prediction)*100)
print('confusion matrix:')
print(confusion_matrix(y_test,y_prediction))
for element in classifier_algorithms:
    model = element[0]
    model.fit(X_sm,y_sm)
    y_prediction = model.predict(x_test)
    print(element[1])
    print('confusion matrix:')
    print(confusion_matrix(y_test,y_prediction))
    print("accuracy:")
    print(accuracy_score(y_test,y_prediction)*100)
    print("recall score:",recall_score(y_test,y_prediction)*100)
    print("precision",precision_score(y_test,y_prediction)*100)
    print("f1_Score",f1_score(y_test,y_prediction)*100)
    print("ROC AUC",roc_auc_score(y_test,y_prediction)*100)
sv = model3.fit(X_sm,y_sm)
pickle.dump(sv, open('wis.pkl', 'wb'))
