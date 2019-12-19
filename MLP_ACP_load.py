import nltk
from nltk.util import ngrams

from gensim.models import FastText as fText
import urllib.request, random

from keras.models import Sequential
from keras.layers import Dense

import csv

from keras.layers import merge, Convolution2D, MaxPooling2D, Input
from keras.layers import Dense, Activation
from keras.models import Sequential, Model

import sys

import numpy as np
#!/usr/bin/python"
# coding: utf-8

from gensim.models import FastText as fText
import urllib.request, random

from keras.models import Sequential
from keras.layers import Dense

import csv

from keras.layers import merge, Convolution2D, MaxPooling2D, Input
from keras.layers import Dense, Activation
from keras.models import Sequential, Model

import sys

import numpy as np


import treetaggerwrapper

import sys
import nltk
import numpy as np
import re
from random import randint

import treetaggerwrapper


def generation_ngram_newVersion(n, f_dataset):
	

	liste_ngrams = []

	pivot = int(n/2)
	
	Fichier_exemple = f_dataset


	tagger = treetaggerwrapper.TreeTagger(TAGLANG='fr', TAGINENC='utf-8',TAGOUTENC='utf-8')	
	F = open(Fichier_exemple, 'r', encoding ='utf-8')
	reader_teste = csv.reader(F, delimiter=';', quoting=csv.QUOTE_ALL, )
	entree = ()
	Liste_entree_negatifs = []

	for idf, Y, X, mp,src, alea in reader_teste:
	#for Y, X, mp in reader_teste:		
		entree = X, Y
		Liste_entree_negatifs.append (entree)

	count = 0
	for e in Liste_entree_negatifs:
		if e[1] == "1" or e[1] == "0" or e[1] == "9":
			#############Teste solution avec Split 
		
			#L_phrase = e[0].split(' ')

			###############teste solution avec treetager tokenisation############
			
			sentence  = e[0]
			sentence = sentence.replace(';','')
			sentence = sentence.replace("'", chr(39))
			sentence = sentence.replace('\'',chr(39))

			sentence = sentence.replace("d\'"," deeee ")
			sentence = sentence.replace("l\'", " leeee ")

			sentence_tagger = tagger.tag_text(sentence)
			tag = treetaggerwrapper.make_tags(sentence_tagger)



			L_phrase = [] 
			for t in tag:
				L_phrase.append(t[0])
				
			#print(L_phrase)
			for e in L_phrase:
				if e == "leeee":
					i = L_phrase.index(e)
					L_phrase[i] = "l\'"
				if e == 'deeee':
					i = L_phrase.index(e)
					L_phrase[i] = "d\'"
			
			print(L_phrase)
			#print("####################################")

			crochetouvrant = e[0].find('[')
			crochetfermant = e[0].find(']')


			index1 = L_phrase.index('[')
			index2 = L_phrase.index(']')
			lemotpivot = index2 - index1 
			#transformation des ESNN composées en une seul position dans la phrase
			esnncompose = ""
			for i in range(lemotpivot-1):
				esnncompose = esnncompose +" "+ L_phrase[lemotpivot+i]

			#listmot[index1+1] = esnncompose
			#del listmot[index1+2:index2-1]
			#fin de transformation des ESNN N


			phrase_ngrame = []

			debut = index1 - pivot
			fin = index2 + pivot

			la_taille= fin-debut+1
			#print(la_taille)
			for i in range(la_taille):
				try:
					phrase_ngrame.append(L_phrase[debut+i])
				except Exception as e:
					#dans le cas ou il n'y a pas assez de mots pour former le ngrams: par exemple dans le cas ou le mot pivot se trouvce au debut
					#page = open ("../corpus/listedesnomfrancais.txt","r")
					page = open ("/corpus/listedesnomfrancais.txt","r")
					liste_mots_francais = page.readlines()
					mot_random = random.choice(liste_mots_francais)
					mot_random = mot_random.rstrip()
					page.close()
					phrase_ngrame.append(mot_random)
				

			liste_ngrams.append(phrase_ngrame)	


	liste_ngrams_sanscrochet = []
	output =  open("NgramsCorpustestMix.txt", "w")
	#output =  open("mix_corpus_test_validation/NgramsCorpusValidationMix.txt", "w")
	for element in liste_ngrams:
		element.remove('[')
		element.remove(']')
		liste_ngrams_sanscrochet.append(element)
		output.write(str(liste_ngrams_sanscrochet)+"\n")


	

	output.close()
		#print (len(liste_ngrams_sanscrochet))
	#print(len(liste_ngrams_sanscrochet))
		





	return liste_ngrams_sanscrochet

	

	

	



		
def vectorisation_new (n, fichier_dataset):

	corpus = generation_ngram_newVersion(n,fichier_dataset)

	#corpusneg = generation_exemple_negatif(n)


	print("la taille du cropus est : "+str(len(corpus)))

 
	page = open ("/corpus/listedesnomfrancais.txt","r")
	liste_mots_francais = page.readlines()

	mot = random.choice(liste_mots_francais)
	mot = mot.rstrip()
	page.close()

	print("chargement du modél Fastext ...")

	fastText_wv = fText.load_fasttext_format("/corpus/cc.fr.300") 

	#matrice = np.zeros((6,3), dtype='int32')

	#fichierpersistant = open("corpus_wikipedia/corpus_negatifs_annotees/vecteurs_corpus514Corigee+wiki_negatifs.txt","w")

	X = np.zeros(shape=(len(corpus),n,300), dtype='float32')
	X = np.delete(X, np.where(X==0))
#X=[]
#vecteur = np.ndarray(shape=(300), dtype='float32')
	#print (len(corpus))
	fichier_mot_non_present_fastext = open ("phrase_mot_non_present_dans_fastext_"+"Corpus_Validation_tokenization"+"_.csv", "w")
	fichier_mot_non_present_fastext.write("Phrase;Mot_non present dans fastext;mot de remplassement \n")
	for phrase in corpus:
		#Traitement des cas des ESNNComposée.
		if len(phrase)>n:
			position = int(n/2)
			#position -> position = 2 si n =5 et position =3 si n=7 ect...

			index_phrase = corpus.index(phrase)

			taille= len(phrase)
			l = phrase[position:taille-position]
			ESNNcomposee = ' '.join(l)
			del phrase[position:taille-position]
			phrase.insert(position,ESNNcomposee)

			#print (phrase)
			corpus[index_phrase] = phrase 
	#print (len(corpus))
	phrase_mot_non_present_dans_fastext = []
	for phrase in corpus:
		matrice = np.zeros((n,300), dtype='float32')
		matrice = np.delete(matrice, np.where(matrice==0))
		#matrice = np.ndarray(shape=(5,300), dtype='float32')
	
		i = 0
		for word in phrase:
			#vecteur = []
		
			try:
				word = word.replace("’","\'")
				vecteur = fastText_wv[str(word)]
				i=i+1
			except Exception as e:
				if word == "D5a":
					mot = "5"
					t = (phrase, word, mot)
				else:
					mot = random.choice(liste_mots_francais)
					mot = mot.rstrip()
					print(mot)
					t = (phrase, word, mot)
				fichier_mot_non_present_fastext.write(str(phrase) +";" + str(word) +";"+ str(mot) +"\n")
				phrase_mot_non_present_dans_fastext.append(t)
				vecteur = fastText_wv[mot]
				i=i+1
		
			matrice = np.append(matrice, vecteur)
		
			#matrice.append(vecteur)

		#m = np.asarray(matrice)

	

		#fichierpersistant.write(str(phrase)+"\n"+str(matrice)+"\n")
		X = np.append(X, matrice)
		#print(i)
		#X.append(m)

	entree = np.asarray(X)



	
	

	fichier_mot_non_present_fastext.close()	

		
	#print(entree.shape)

	#print (len(matrice))

	#print(len(X))

	entree_sans_zero = np.delete(X, np.where(X==0))


	print(entree_sans_zero.shape)

	entree_sans_zero_reshape = np.reshape(entree_sans_zero, (len(corpus),n,300))

	print (entree_sans_zero_reshape.shape)
	print (type(entree_sans_zero_reshape))
	



	return (entree_sans_zero_reshape)




def extrect_Y(dataset_X):
	F = open(dataset_X, 'r', encoding ='utf-8')
	reader_teste = csv.reader(F, delimiter=';', quoting=csv.QUOTE_ALL, )
	
	Y = np.zeros((1), dtype='int32')
	Y = np.delete(Y, np.where(Y==0))

	for idf, annotation, X, mp,src, alea in reader_teste:
		if int(annotation) == 0:
			
			Y = np.append(Y, int(annotation))
		if int(annotation) == 1:
			
			Y = np.append(Y, int(annotation))
	
	return (Y)



	
def load_RF(dataset_X):
	

	from keras.models import Sequential
	from keras.layers.core import Dense, Dropout, Activation
	from keras.layers.embeddings import Embedding
	from keras.layers.recurrent import GRU

	from keras.layers import Flatten, Dense
	import keras
	from keras.utils import np_utils
	from keras.layers import Dropout
	from sklearn.model_selection import StratifiedKFold
	import matplotlib.pyplot as plt
	from keras.utils import plot_model

	from joblib import dump, load
	from sklearn.decomposition import PCA

	from keras.models import load_model
	
	print('Vectorisation of inputs..... \n')
	print (sys.argv)
	print(sys.argv[1])

	taille_ngrams = int(sys.argv[1])

	#X = vectorisation (int(sys.argv[1]), dataset_X)
	X = vectorisation_new(int(sys.argv[1]), dataset_X)

	taille_vecteur_phrase = int(sys.argv[1]) * 300

	print('Creat the outputs ... \n')

	Y = extrect_Y(dataset_X)
	import numpy
	# fix random seed for reproducibility
	seed = 1
	numpy.random.seed(seed)

	if len(X)== len(Y):
		print("X et Y taille identique")
	else:
		print("*********-*********-************-*ATTENTION X ET Y NE SONT PAS DE LA MÉME TAILLE VEUILLEZ VERIFIER VOS ENTREEE*******************************************")


	X_validation = X
	
	X_validation = np.reshape(X_validation , ( int(len(X_validation)) , taille_vecteur_phrase) )

	
	Y_validation = Y

	#c'est juste pour faire l'acp
	
	

	#X_train_for_pca = vectorisation_new(int(sys.argv[1]), "/home/medad/methode_par_apprentissage/resultats_apprentissage_ models Fevrier/Validation_experimentation_juillet/corpus/corpus_apprentissage_fevrier_teste_acp.csv")
	X_train_for_pca = vectorisation_new(int(sys.argv[1]), "/corpus/corpus_apprentissage_fevrier_teste_acp.csv")
	#pour 1 grams
#	X_train_for_pca = np.reshape(X_train_for_pca, (int(len(X_train_for_pca)), taille_vecteur_phrase) )
	
	#pour 5grams
	X_train_for_pca = np.reshape(X_train_for_pca, (568, taille_vecteur_phrase) )

	#pour 1 et 5grams (pour 5 grams j'ai change la ligne juste au dessus 466)
	#pca = PCA(0.99)
	
	#Pour 7grams
	#pca = PCA(n_components = 368)
	#pca.fit(X_train)
	#X_train_PCA = pca.fit_transform(X_train_for_pca, y=None)

	#print("X_train_PCA.shape : ", str(X_train_PCA.shape))
	
	#X_test_PCA = pca.transform(X_validation)

	#print("X_test_PCA.shape : ", str(X_test_PCA.shape))

	print('charger le modéle :  ... \n')

	

	chemin_model = '/MLP_ACP_models/'

	if taille_ngrams == 1 :

		# print('---------------------------------------\n----------------------------------------\n\ncharger le modéle 1 grams:  ... \n\n-----------------------------\n\n')

		# config = 'MLP_PCA_relu_relu_sigmoid_29_5_adam_dynamique_0.0_1'
		
		
		# m = load_model(chemin_model+config+'.h5')
				
		# print('\n Resultats du teste du model MLP_ACP en utilisant le corpus de teste\n')			
	
		# acc_validation = predire(m, X_test_PCA, Y_validation,config)


		import os
		for file in os.listdir(chemin_model):
			if file.endswith("1.h5"):
				clf = load_model(chemin_model+file)
				acc_Validation = predire(clf, X_test_PCA, Y_validation,file)


	if taille_ngrams == 5 :

		# print('---------------------------------------\n----------------------------------------\n\ncharger le modéle 5 grams:  ... \n\n-----------------------------\n\n')
		
		# config = 'MLP_PCA_relu_relu_sigmoid_32_6_adam_dynamique_0.5_5'

		
		# m = load_model(chemin_model+config+'.h5')
				
		# print('\n Resultats du teste du model MLP_ACP en utilisant le corpus de teste\n')			
	
		# acc_validation = predire(m, X_test_PCA, Y_validation,config)


		import os
		for file in os.listdir(chemin_model):
			if file.endswith("5.h5"):
				clf = load_model(chemin_model+file)
				acc_Validation = predire(clf, X_test_PCA, Y_validation,file)


	if taille_ngrams == 7 :

		# print('---------------------------------------\n----------------------------------------\n\ncharger le modéle 7 grams:  ... \n\n-----------------------------\n\n')

		# config = 'MLP_PCA_relu_elu_sigmoid_61_12_adam_dynamique_0.5_7'

		
		# m = load_model(chemin_model+config+'.h5')
				
		# print('\n Resultats du teste du model MLP_ACP en utilisant le corpus de teste\n')			
	
		# acc_validation = predire(m, X_test_PCA, Y_validation,config)
		# import os
		# for file in os.listdir(chemin_model):
		# 	if file.endswith("7.h5"):
		# 		clf = load_model(chemin_model+file)
		# 		acc_Validation = predire(clf, X_test_PCA, Y_validation,file)





		import os
		liste_file = os.listdir(chemin_model)
		# idx = 0 
		# thiselem = liste_file[idx]
		# taile_liste = len (liste_file)
		# idx = (idx + 1) % len(liste_file)
		#nextelem = liste_file[idx]
		# liste_file.remove('MLP_PCA_softplus_relu_sigmoid_122_24_adam_dynamique_0.5_7.h5')
		# liste_file.remove('MLP_PCA_softplus_relu_sigmoid_61_12_adam_dynamique_0.5_7.h5')


		# liste_file.remove('MLP_PCA_softplus_softplus_sigmoid_61_12_adam_dynamique_0.3_7.h5')

		# liste_file.remove('MLP_PCA_relu_elu_sigmoid_61_12_adam_dynamique_0.3_7.h5')
		

		
		for file in liste_file:
			#liste_file.remove('MLP_PCA_softplus_relu_sigmoid_122_24_adam_dynamique_0.5_7.h5')
			if file.endswith("7.h5"):
				
				try:
					#Pour 7grams
					pca = PCA(n_components = 368)
					#pca.fit(X_train)
					X_train_PCA = pca.fit_transform(X_train_for_pca, y=None)

					print("X_train_PCA.shape : ", str(X_train_PCA.shape))
	
					X_test_PCA = pca.transform(X_validation)
					
					clf = load_model(chemin_model+file)
					acc_Validation = predire(clf, X_test_PCA, Y_validation,file)
				except Exception as e:
					#raise e
					pca = PCA(n_components = 369)
					#pca.fit(X_train)
					X_train_PCA = pca.fit_transform(X_train_for_pca, y=None)

					print("X_train_PCA.shape : ", str(X_train_PCA.shape))
	
					X_test_PCA = pca.transform(X_validation)
					
					clf = load_model(chemin_model+file)
					acc_Validation = predire(clf, X_test_PCA, Y_validation,file)

				










def predire(model,X_test, Y_test, execution):
	
	print('#######################################  Résultat des PREDICTIONS sur le corpus de teste ################################################ \n')
	print('CONFIG : ' + execution)
	# calculate predictions
	predictions = model.predict(X_test)
	# calculate predictions
	

	#print(predictions)




	
	# round predictions
	rounded = [round(x[0]) for x in predictions]



	from pandas_ml import ConfusionMatrix

	#print(Y_test)

	cm = ConfusionMatrix(Y_test, rounded)
	cm.print_stats()
	

	print(rounded)
	print(Y_test)

	correct = 0
	for i in range(len(rounded)):
		if int(rounded[i]) == Y_test[i]:
			correct = correct +1
	
	p = (correct / i)*100

	print("le taux exactitude (nombre d'ESNN reconu sur le nombre total ESNN) : " + str(p))
	return p

print ("*******************Validation MLP_ACP ***************************\n")



#Emergence 93
#load_RF("corpus/corpus_emergence_AM_ancien.csv")

#corpus_Validation_mix
load_RF("/corpus/corpus_validation_mix.csv")

