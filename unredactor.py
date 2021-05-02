import glob
import io
import os
import pdb
import sys
import numpy as np
import fnmatch
import re
from sklearn.linear_model import SGDClassifier
import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk

from commonregex import CommonRegex
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
import glob
import io
import os
import pdb
import sys

import re
import nltk
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk import ne_chunk

import random
import numpy as np
from sklearn.feature_extraction import DictVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn import svm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import ensemble
from sklearn.metrics.pairwise import cosine_similarity
import regex

def Read_files(text_files):
    # print(text_files)
    data = []
    filenames =[]
    for filename in glob.glob(os.path.join(os.getcwd(),text_files)):
        #print(filename)
        filenames.append(filename.split('/')[-1])
        print(filenames)
        with open(os.path.join(os.getcwd(), filename), "r", encoding='utf-8') as f:
            data1 = f.read()
            data.append(data1)
    return data, filenames

def read_files(path):
    os.chdir(path)
    data = []
    file_names = []
    for file in os.listdir():
        if file.endswith(".txt"):
            file_names.append(file)
            f_path=f'{path}\{file}'
            with io.open(f_path, 'r', encoding='utf-8') as file1:
                text = file1.read()
                data.append(text)
            
    return data, file_names

def get_redacted_entity(data):
    person_list=[]
    #person_list1=[]
    for sent in sent_tokenize(data):
        from nltk import word_tokenize
        x=word_tokenize(sent)
        for chunk in ne_chunk(pos_tag(x)):
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':       
                a=""
                for c in chunk.leaves():
                    a=a+c[0]
                    a=a+" "
                person_list.append(a[:-1])
    count=len(person_list)            
    personlist1=set(person_list)
    person_list1=list(personlist1)
    #print(persons)
    person_list1=sorted(person_list1, reverse= True)
    #print(person_list1)
    return person_list1

def retrieve_train_features(text, person_name_list):
    features = []
    cc = len(text)
    wc = len(text.split())
    sc = len(sent_tokenize(text))
    cs = 0
    
    for i in text:
        if i == " ":
            cs+=1
    
    for i in range(0, len(person_name_list)):
        dict = {}
        dict['sent_count'] = sc
        dict['word_count'] = wc
        dict['character_count'] = cc
        dict['space_count'] = cs
        dict['name_length'] = len(person_name_list[i])
        dict['total_names'] = len(person_name_list)
        
        features.append(dict)

    return features

def retrieve_test_features(text, redacted_names_in_block):
    features = []
    cc = len(text)
    wc = len(text.split())
    sc = len(sent_tokenize(text))
    cs = 0
    
    for i in text:
        if i == " ":
            cs+=1
            
    for i in range(0, len(redacted_names_in_block)):
        dict = {}
        dict['sent_count'] = sc
        dict['word_count'] = wc
        dict['character_count'] = cc
        dict['space_count'] = cs
        dict['name_length'] = len(redacted_names_in_block[i])
        dict['total_names'] = len(redacted_names_in_block)
        
        features.append(dict)

    return features

def Fields_to_redact(person_list1):
    replace=[]
    for element in person_list1:
        replace.append(element)
    return replace

def Redact(replace,data):
    for j in range(0,len(replace)):
        if replace[j] in data:
            length = len(replace[j])
            data = re.sub(replace[j], length*'\u2588', data, 1)
    return data

def Get_Unique_Names(names_list):
    names_list_unique = (set(names_list))
    names_list_unique = list(names_list_unique)
    #unique = [i for n, i in enumerate(names_list) if i not in names_list[:n]]
    #unique_namelist=unique
    return names_list_unique 
  
def Save_to_output_redacted(redact_result, folder, file_name):
    new_file = file_name.replace(".txt", ".redacted.txt")
    isFolder = os.path.isdir(folder)
    if isFolder== False:
        os.makedirs(os.path.dirname(folder))
    with open( os.path.join(folder, new_file), "w+", encoding="utf-8") as f:
        f.write(redact_result)
        f.close()
        
def Save_to_output_predicted(redact_result, folder, file_name, data_list, redacted_names):
    isFolder = os.path.isdir(folder)
    if isFolder== False:
        os.makedirs(os.path.dirname(folder))
    result = Get_predicted_output(redact_result, data_list, redacted_names)
    with open( os.path.join(folder, file_name), "w+", encoding="utf-8") as f:
        f.write(result)
        f.close()
        
def Get_predicted_output(redact_result, data_list, redacted_names):
    result =redact_result
    for i in range(0, len(data_list)):
        names = ""
        for j in data_list[i]:
            names += j
            names += ","
        names = names[:-1]
        result +="\n {} top 5` predicted names are {}".format(redacted_names[i], names) 
    return result
  

def Read_files2(text_files):
    # print(text_files)
    print("123")
    data = []
    filenames =[]
    for filename in glob.glob(os.getcwd()):
        print(filename)
        print(filename)
        filenames.append(filename.split('/')[-1])
        print(filenames)
        with open(os.path.join(os.getcwd(), filename), "r") as f:
            data1 = f.read()
            data.append(data1)
    return data, filenames
  
  
def retrieve_predicted_words(probabilities_all_classes, Names_Redacted):
    All_predicted_words_review = []
    for test_word in range(0, len(Names_Redacted)):
        test_word_probabilities = probabilities_all_classes[test_word]
        top_5_idx = np.argsort(test_word_probabilities)[-5:]
        #print(top_5_idx)
        predicted_words = []
        for i in range(0,5):
            index_range = top_5_idx[i]
            predicted_word = names_unique[index_range]
            predicted_words.append(predicted_word)
        #print(predicted_words)
        All_predicted_words_review.append(predicted_words)
    #print(All_predicted_words_review)
    return (All_predicted_words_review)

if __name__=='__main__':
#train the model
    input_path = "input"
    output_path_redacted = "redacted"
    output_path_prediction = "predicted"
    train_data, file_names = Read_files(input_path)
    replace_result_list = []
    names_list = []
    redacted_data_list=[]
    redacted_data=[]
    full_list_training_features = []
    full_list_names = []
    redacted_result = []
    for itr in range(0, len(train_data)):
    
        person_list_result = get_redacted_entity(train_data[itr])
        #print(person_list_result)
        replace_result = Fields_to_redact(person_list_result)
    
        for entry in replace_result_list:
            for names in entry:
                names_list.append(names)
    
        redact_result = Redact(replace_result,train_data[itr])
        Save_to_output_redacted(redact_result, output_path_redacted, file_names[itr])
        redacted_data_list.append(redact_result)
        
        list_names_dict_features = retrieve_train_features(train_data[itr], person_list_result)
        full_list_training_features.extend(list_names_dict_features)
    
        full_list_names.extend(person_list_result)
        
    #TODO: store the results in the file
    #for now I will store the redacted result in a list and pass it in the testing function
    
    #print(redact_result)

    v = DictVectorizer()
    X = v.fit_transform(full_list_training_features).toarray()
    full_list_names = np.array(full_list_names)
    model = svm.SVC(probability=True)
    #model = SGDClassifier()
    model.fit(X, full_list_names)
    names_unique = Get_Unique_Names(full_list_names)

    #redacted_data = redacted_data_list[:12]
    #read redacted data from path
    redacted_data, file_names = read_files(output_path_redacted)
    
    for i in range(0, 12):
        
        redacted_names = re.findall(r'(\u2588+)', redacted_data[i])
        test_features = retrieve_test_features(redacted_data[i], redacted_names)
        if len(test_features) > 0:
            X_test = v.fit_transform(test_features).toarray()
            probabilities_all_classes = model.predict_proba(X_test)
            All_predicted_words_review = retrieve_predicted_words(probabilities_all_classes, redacted_names)
            Save_to_output_predicted(redacted_data[i], output_path_prediction, file_names[i], All_predicted_words_review, redacted_names)
            
