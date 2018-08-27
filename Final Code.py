import pandas
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
import csv
import os
from sklearn.feature_extraction.text import CountVectorizer
import re

#################################################################################################
# Opening the file and inputting data
#################################################################################################
path="A:\\"
dirc = "zoo_data1.csv"
with open(os.path.join(path ,dirc ), newline='\n') as f:
    reader = csv.reader(f)
    Animal_names = [str(row[0]) for row in reader]

with open(os.path.join(path ,dirc ), newline='\n') as f:
    reader = csv.reader(f)
    Animal_types = [str(row[-1]) for row in reader]
    
with open(os.path.join(path ,dirc ), newline='\n') as f:
    reader = csv.reader(f)
    Animal_features = [str(row[1:-2]) for row in reader]

count_vect = CountVectorizer (lowercase=False, analyzer='word')

Bag_of_words = count_vect.fit_transform(Animal_features)
dataset_without_names=Bag_of_words.toarray()
Bag_of_words = pandas.DataFrame.from_dict(Bag_of_words.toarray())


Feature_names = count_vect.get_feature_names()
Feature_names = pandas.DataFrame.from_dict(Feature_names)

#changing types in text to numbers
for i in range(0,len(Animal_types)):
    if Animal_types[i]=='one':
        Animal_types[i]=1
    elif Animal_types[i]=='two':
        Animal_types[i]=2
    elif Animal_types[i]=='three':
        Animal_types[i]=3
    elif Animal_types[i]=='four':
        Animal_types[i]=4
    elif Animal_types[i]=='five':
        Animal_types[i]=5
    elif Animal_types[i]=='six':
        Animal_types[i]=6
    elif Animal_types[i]=='seven':
        Animal_types[i]=7
Anames=Animal_names        
Animal_types = pandas.DataFrame.from_dict(Animal_types).astype(int)
Animal_names = pandas.DataFrame.from_dict(Animal_names)
Animal_names.columns=['Animal_names']

Bag_of_words['Animal_names']=Animal_names    

###############################################################################################################
#Train Test Split
###############################################################################################################

zoo_target = Animal_types.values
zoo_data = Bag_of_words.values

validation_size  = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(zoo_data, zoo_target, test_size=validation_size, random_state=7)

y = Y_train.ravel()
Y_train = np.array(y).astype(int)
Y_train_CNN = to_categorical(Y_train)
Y_train_CNN = np.delete(Y_train_CNN,0,1)
X_train1 = np.delete(X_train,-1,1)
X_validation1 = np.delete(X_validation,-1,1)

################################################################################################################
# CNN model
################################################################################################################


def get_cnn_model():    # added filter
    model = Sequential()
    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    # 1000 is num_max
    model.add(Embedding(len(X_train1[1,:]),
                        20,
                        input_length=len(X_train1[1,:])))
    model.add(Dropout(0.2))
    model.add(Conv1D(128,
                     3,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Dropout(0.2))
    model.add(Activation('sigmoid'))
    #model.add(Dense(7))
    #model.add(Dropout(0.5))
    #model.add(Activation('sigmoid'))
    model.add(Dense(7))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
    return model

m = get_cnn_model()


################################################################################################################
# Machine Learning Model Fitting
################################################################################################################

model1 = LogisticRegression(class_weight='balanced')
model2 = KNeighborsClassifier()
model3 = RandomForestClassifier(class_weight='balanced')
model4 = SVC(C=5.0,class_weight='balanced',verbose=True, probability=True)
model5 = get_cnn_model()

model1.fit(X_train1 , Y_train)
model2.fit(X_train1 , Y_train)
model3.fit(X_train1 , Y_train)
model4.fit(X_train1 , Y_train)
model5.fit(X_train1 , Y_train_CNN ,batch_size=32,epochs=200,verbose=1,validation_split=0.2)
################################################################################################################
# Results to the fitting 
################################################################################################################
y_val = Y_validation.ravel()
Y_Eval_CNN = np.array(y_val).astype(int)
Y_Eval_CNN = to_categorical(Y_Eval_CNN)
Y_Eval_CNN = np.delete(Y_Eval_CNN,0,1)

score1 = model1.score(X_validation1,Y_validation)
score2 = model2.score(X_validation1,Y_validation)
score3 = model3.score(X_validation1,Y_validation)
score4 = model4.score(X_validation1,Y_validation)
score5 = model5.evaluate(X_validation1,Y_Eval_CNN, batch_size=32)[1]

result1 = model1.predict(X_validation1)
result1 = result1.astype(str)
result2 = model2.predict(X_validation1)
result2 = result2.astype(str)
result3 = model3.predict(X_validation1)
result3 = result3.astype(str)
result4 = model4.predict(X_validation1)
result4 = result4.astype(str)
result5 = model5.predict(X_validation1)
result5 = np.argmax(result5,axis=1)+1
result5 = result5.astype(str)

results_prob1=model1.predict_proba(X_validation1)*100
results_prob1=results_prob1.astype(int)
results_prob1=results_prob1.astype(str)
results_prob2=model2.predict_proba(X_validation1)*100
results_prob2=results_prob2.astype(int)
results_prob2=results_prob2.astype(str)
results_prob3=model3.predict_proba(X_validation1)*100
results_prob3=results_prob3.astype(int)
results_prob3=results_prob3.astype(str)
results_prob4=model4.predict_proba(X_validation1)*100
results_prob4=results_prob4.astype(int)
results_prob4=results_prob4.astype(str)
results_prob5=model5.predict(X_validation1)*100
results_prob5=results_prob5.astype(int)
results_prob5=results_prob5.astype(str)

results_prob1=np.insert(results_prob1, 0 , X_validation[...,-1],axis=1)
results_prob1=np.insert(results_prob1, 8 , Y_validation[...,0] ,axis=1)
results_prob1=np.insert(results_prob1, 9 , result1 ,axis=1)
results_prob1=np.insert(results_prob1, 0 , ('Animal Name','(1)Mammal','(2)Bird','(3)Reptile','(4)Fish','(5)Amphibian','(6)Bug','(7)Invertebrate','Req_Class','Predicted_Class'),axis=0)

results_prob2=np.insert(results_prob2, 0 , X_validation[...,-1],axis=1)
results_prob2=np.insert(results_prob2, 8 , Y_validation[...,0] ,axis=1)
results_prob2=np.insert(results_prob2, 9 , result2 ,axis=1)
results_prob2=np.insert(results_prob2, 0 , ('Animal Name','(1)Mammal','(2)Bird','(3)Reptile','(4)Fish','(5)Amphibian','(6)Bug','(7)Invertebrate','Req_Class','Predicted_Class'),axis=0)

results_prob3=np.insert(results_prob3, 0 , X_validation[...,-1],axis=1)
results_prob3=np.insert(results_prob3, 8 , Y_validation[...,0] ,axis=1)
results_prob3=np.insert(results_prob3, 9 , result3, axis=1)
results_prob3=np.insert(results_prob3, 0 , ('Animal Name','(1)Mammal','(2)Bird','(3)Reptile','(4)Fish','(5)Amphibian','(6)Bug','(7)Invertebrate','Req_Class','Predicted_Class'),axis=0)

results_prob4=np.insert(results_prob4, 0 , X_validation[...,-1],axis=1)
results_prob4=np.insert(results_prob4, 8 , Y_validation[...,0] ,axis=1)
results_prob4=np.insert(results_prob4, 9 , result4 ,axis=1)
results_prob4=np.insert(results_prob4, 0 , ('Animal Name','(1)Mammal','(2)Bird','(3)Reptile','(4)Fish','(5)Amphibian','(6)Bug','(7)Invertebrate','Req_Class','Predicted_Class'),axis=0)

results_prob5=np.insert(results_prob5, 0 , X_validation[...,-1],axis=1)
results_prob5=np.insert(results_prob5, 8 , Y_validation[...,0] ,axis=1)
results_prob5=np.insert(results_prob5, 9 , result5 ,axis=1)
results_prob5=np.insert(results_prob5, 0 , ('Animal Name','(1)Mammal','(2)Bird','(3)Reptile','(4)Fish','(5)Amphibian','(6)Bug','(7)Invertebrate','Req_Class','Predicted_Class'),axis=0)

print("\n")


sc = [score1,score2,score3,score4,score5]
max_score_ML_model = max (sc)
max_index = sc.index(max_score_ML_model)
sel_model=''
if(max_index==0):
    sel_model='Logistic Regression'
elif(max_index==1):
    sel_model='K-Nearest Neighbours'
elif(max_index==2):
    sel_model='Random Forest'
elif(max_index==3):
    sel_model='Support Vector Machine'
elif(max_index==4):
    sel_model='C Neural Network'

    
print ("The max score is :"+str(max_score_ML_model))
print ("The Selected Model for prediction is : " + sel_model)
print("The scores for each model are as follows :\n")
print("Model Logistic Regression : "+str(score1) )
print("Model K nearest Neighbours : "+str(score2) )
print("Model Random Forest : "+str(score3) )
print("Model Support Vector Machine : "+str(score4) )
print("Model CNN: "+str(score5) )

################################################################################################################
# Printing into Files
################################################################################################################


print("\n--The probablities of classes in respective models and their accuracies are outputted to respective .csv files -- \n")
np.savetxt((path+"LogRegressionresult1.csv"), results_prob1, delimiter=",", newline='\n', fmt ='% 20s',header='Model Name : LOGISTIC REGRESSION', footer='Model Accuracy = '+str(score1))
np.savetxt((path+"KNNresult2.csv"), results_prob2, delimiter=",", newline='\n', fmt ='% 10s',header='Model Name : K-NEAREST NEIGHBOURS', footer='Model Accuracy = '+str(score2))
np.savetxt((path+"RandomForestresult3.csv"), results_prob3, delimiter=",", newline='\n', fmt ='% 10s',header='Model Name : RANDOM FOREST', footer='Model Accuracy = '+str(score3))
np.savetxt((path+"SVMresult4.csv"), results_prob4, delimiter=",", newline='\n', fmt ='% 10s',header='Model Name : SUPPORT VECTOR MACHINE', footer='Model Accuracy = '+str(score4))
np.savetxt((path+"CNNresult5.csv"), results_prob5, delimiter=",", newline='\n', fmt ='% 10s',header='Model Name : CONVOLUTIONAL NEURAL NETWORK', footer='Model Accuracy = '+str(score5))


print("\nProbability of classes in model Logistic Regression : \n")
print(results_prob1)
print("\nProbability of classes in model K-nearest Neighbours : \n")
print(results_prob2)
print("\nProbability of classes in model Random Forest : \n")
print(results_prob3)
print("\nProbability of classes in model Support Vector Machine: \n")
print(results_prob4)
print("\nProbability of classes in model CNN: \n")
print(results_prob5)
################################################################################################################
# Prediction of Type from command Line
################################################################################################################

ans='y'
Feature_names=Feature_names.astype(str)

while(ans=='y' or ans=='Y'):
    print(" ")
    print("Please enter the Attributes of the animal : ")
    print("Choose from the following attributes:\n [ hair, feathers, eggs, milk, airborne, aquatic, predator, toothed, backbone, breathes, venomous, fins, legs, tail, domestic, catsize, type ]")
    x=input();
    
    #Splitting input into list
    
    wordList = re.sub("[^\w]", " ", x).split()
    x_input=[0]*len(Feature_names )
    m=len(wordList)
    n=len(Feature_names)
    for i in range(0,n):
        x_input[i]=0
    for i in range(0,n):
        for j in range(0,m):
            
            if((Feature_names[0][i]==wordList[j])):
                x_input[i]=1
                break
    x_input=np.asarray(x_input)
    input_data=x_input
    x_input=x_input.reshape(1,-1)
   
    Predicted_class=''    
        #Prediction of Type of the given input
    if(max_index==0):
        Predicted_class=(model1.predict(x_input))
        prob=(model1.predict_proba(x_input)*100)
    elif(max_index==1):
        Predicted_class=(model2.predict(x_input))
        prob=(model2.predict_proba(x_input)*100)
    elif(max_index==2):
        Predicted_class=(model3.predict(x_input))
        prob=(model3.predict_proba(x_input)*100)
    elif(max_index==3):
        Predicted_class=(model4.predict(x_input))
        prob=(model4.predict_proba(x_input)*100)
    elif(max_index==4):
        Predicted_class=(model5.predict(x_input))
        Predicted_class=np.argmax(Predicted_class,axis=1)+1
        prob=(model5.predict(x_input)*100)
   
    
    if(Predicted_class[0]==1):
        print("The predicted type is : " +'(1)- Mammal')
        print("\nThe distribution of type probabilities is :")
        print(str(prob))
    elif(Predicted_class[0]==2):
        print("The predicted type is : " + '(2)- Bird')
        print("\nThe distribution of type probabilities is :")
        print(str(prob))
    elif(Predicted_class[0]==3):
        print("The predicted type is : " + '(3)- Reptile')
        print("\nThe distribution of type probabilities is :")
        print(str(prob))    
    elif(Predicted_class[0]==4):
        print("The predicted type is : " + '(4)- Fish')
        print("\nThe distribution of type probabilities is :")
        print(str(prob))
    elif(Predicted_class[0]==5):
        print("The predicted type is : " + '(5)- Amphibian')
        print("\nThe distribution of type probabilities is :")
        print(str(prob))
    elif(Predicted_class[0]==6):
        print("The predicted type is : " + '(6)- Bug')
        print("\nThe distribution of type probabilities is :")
        print(str(prob))
    elif(Predicted_class[0]==7):
        print("The predicted type is : " + '(7)- Invertebrate')
        print("\nThe distribution of type probabilities is :")
        print(str(prob))
    max_score = 0
    animal='x'
    percent=0
    index =-1
    Attributes_in_input_data=0
    #print(input_data[:])
    Animals_in_class=''
    for x in input_data:
        if x==1 :
            Attributes_in_input_data+=1
    print("\n")
    for i in range (0,len(zoo_target)):
        if((zoo_target[i]==Predicted_class[0])[0]):
            
            Attributes_in_item_ds=0
            for x in dataset_without_names[i]:
                if x==1:
                    Attributes_in_item_ds+=1
            
            Animals_in_class+= Anames[i]+' '
            temp=0
            
            for j in range(0,len(input_data)):
                if(input_data[j]==1 and (dataset_without_names[i][j]==input_data[j])):
                    temp+=1
                    
            if(temp+(temp/Attributes_in_item_ds)>max_score):
                max_score=temp+(temp/Attributes_in_item_ds)
                index=i
                percent = (temp/Attributes_in_input_data)*100
    
    print ("All the animals of this type are :")
    print (Animals_in_class)
    print('')
    print ("Out off all of these animals the closest match to the given attributes in the predicted type is : "+Anames[index])
    print('')
    print("Percentage of match with the animal is : "+str(percent))
    print('')
    print("Do you want to continue(y/n): ")
    ans=input();

################################################################################################################
# End of Program
################################################################################################################

