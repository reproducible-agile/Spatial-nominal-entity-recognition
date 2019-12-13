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


# Fichier_exemple = "corpus_wikipedia/corpus_negatifs_annotees/corpus_wiki_negatifs_annote.csv"

# ficier_negatifs_netoyer = open("corpus_wikipedia/corpus_negatifs_annotees/fichier_negatifs_wiki_netoyer.txt","w")

# tagger = treetaggerwrapper.TreeTagger(TAGLANG='fr', TAGINENC='utf-8',TAGOUTENC='utf-8')	
# F = open(Fichier_exemple, 'r')
# reader_teste = csv.reader(F, delimiter=';', quoting=csv.QUOTE_NONE, )
# entree = ()
# Liste_entree_negatifs = []

# for Y,mot_lexique, X in reader_teste:
# 	entree = X, Y, mot_lexique
# 	Liste_entree_negatifs.append (entree)

# count = 0
# for e in Liste_entree_negatifs:
# 	if e[1] == "1" or e[1] == "0" or e[1] == "9":
# 		#############Teste solution avec Split 
	
# 		#L_phrase = e[0].split(' ')

# 		###############teste solution avec treetager tokenisation############
		
# 		sentence  = e[0]
# 		sentence = sentence.replace(';','')
# 		sentence = sentence.replace("'", chr(39))
# 		sentence = sentence.replace('\'',chr(39))
# 		sentence = sentence.replace("d\'"," deeee ")
# 		sentence = sentence.replace("l\'", " leeee ")

# 		sentence_tagger = tagger.tag_text(sentence)
# 		tag = treetaggerwrapper.make_tags(sentence_tagger)



# 		L_phrase = [] 
# 		for t in tag:
# 			L_phrase.append(t[0])
			
# 		print(L_phrase)
# 		for e in L_phrase:
# 			if e == "leeee":
# 				i = L_phrase.index(e)
# 				L_phrase[i] = "l\'"
# 			if e == 'deeee':
# 				i = L_phrase.index(e)
# 				L_phrase[i] = "d\'"
		
# 		print(L_phrase)
# 		fichier_negatifs_wiki_netoyer.write(L_phrase)

# 		print("####################################")

# 		crochetouvrant = e[0].find('[')
# 		crochetfermant = e[0].find(']')
# 		phrase = e[0]



		


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
					page = open ("/home/medad/methode_par_apprentissage/corpus/listedesnomfrancais.txt","r")
					liste_mots_francais = page.readlines()
					mot_random = random.choice(liste_mots_francais)
					mot_random = mot_random.rstrip()
					page.close()
					phrase_ngrame.append(mot_random)
				

			liste_ngrams.append(phrase_ngrame)	


	liste_ngrams_sanscrochet = []
	#output =  open("mix_corpus_test_validation/NgramsCorpustestMix.txt", "w")
	output =  open("mix_corpus_test_validation/NgramsCorpusValidationMix.txt", "w")
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

 
	page = open ("/home/medad/methode_par_apprentissage/corpus/listedesnomfrancais.txt","r")
	liste_mots_francais = page.readlines()

	mot = random.choice(liste_mots_francais)
	mot = mot.rstrip()
	page.close()

	print("chargement du modél Fastext ...")

	fastText_wv = fText.load_fasttext_format("/home/medad/methode_par_apprentissage/corpus/cc.fr.300") 

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



def extrect_Y(dataset_Y):
	#tokenizer = nltk.data.load('tokenizers/punkt/PY3/french.pickle')

	#fichier = open("corpus/Y_negatif_duplique_general.txt", "r")
	fichier = open(dataset_Y, "r")
	fichierlu = fichier.read()
	Y = np.zeros((1), dtype='int32')
	Y = np.delete(Y, np.where(Y==0))
	print(len(Y))

	
	i = 0
	positif = 0
	zero = 0
	neuf = 0
	for element in fichierlu:
		element = element.strip()
		if element.isdigit():

			element = int(element)
			if element == 9:
				element = 0
			#print(element+element)
			Y = np.append(Y,element)
			if (element == 1):
				positif = positif+1
			if (element == 0 ):
				zero= zero +1
			if (element == 9):
				neuf = neuf +1
			i = i+1


	
	

	return (Y)




def load_RF(dataset_X,dataset_Y):
	
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

	from keras.models import load_model
	
	print('Vectorisation of inputs..... \n')
	print (sys.argv)
	print(sys.argv[1])

	taille_ngrams = int(sys.argv[1])

	#X = vectorisation (int(sys.argv[1]), dataset_X)
	X = vectorisation_new(int(sys.argv[1]), dataset_X)
	print('Creat the outputs ... \n')

	Y = extrect_Y(dataset_Y)
	import numpy
	# fix random seed for reproducibility
	seed = 1
	numpy.random.seed(seed)

	if len(X)== len(Y):
		print("X et Y taille identique")
	else:
		print("*********-*********-************-*ATTENTION X ET Y NE SONT PAS DE LA MÉME TAILLE VEUILLEZ VERIFIER VOS ENTREEE*******************************************")



	X_validation = X
	X_validation = np.reshape(X_validation, (int(len(X_validation)), int(sys.argv[1])*300))

	Y_validation = Y




	

	if taille_ngrams == 1 :

		print('---------------------------------------\n----------------------------------------\n\ncharger le modéle 1 grams:  ... \n\n-----------------------------\n\n')

		
#traitement_Best_selon_test

		# config = 'RF_max_depth : 22_criter : entropy_n_arbre_by_forest : 5001grams'
				
		# chemin_model = "/home/medad/methode_par_apprentissage/resultats_apprentissage_ models Fevrier/RF/"
		# clf = load(chemin_model+config+'.joblib') 		
			
		# print('\n Resultats du teste du model RF en utilisant le corpus de validation\n')			
	
		# acc_test_corpus_test = predire(clf, X_validation, Y_validation ,config)
#traitement ALL
		chemin_model = "/home/medad/methode_par_apprentissage/resultats_apprentissage_ models Fevrier/RF/"
		import os
		for file in os.listdir(chemin_model):
			if file.endswith("1grams.joblib"):
				clf = load(chemin_model+file)
				acc_Validation = predire(clf, X_validation, Y_validation,file)

	if taille_ngrams == 5 :

#traitement_Best_selon_test
		print('---------------------------------------\n----------------------------------------\n\ncharger le modéle 5 grams:  ... \n\n-----------------------------\n\n')
		
		# config = 'RF_max_depth : 46_criter : gini_n_arbre_by_forest : 505grams'
				
		# chemin_model = "/home/medad/methode_par_apprentissage/resultats_apprentissage_ models Fevrier/RF/"
		# clf = load(chemin_model+config+'.joblib') 		
			
		# print('\n Resultats du teste du model RF en utilisant le corpus de validation\n')			
	
		# acc_test_corpus_test = predire(clf, X_validation, Y_validation ,config)

#traitement ALL
		chemin_model = "/home/medad/methode_par_apprentissage/resultats_apprentissage_ models Fevrier/RF/"
		import os
		for file in os.listdir(chemin_model):
			if file.endswith("5grams.joblib"):
				clf = load(chemin_model+file)
				acc_Validation = predire(clf, X_validation, Y_validation,file)


	if taille_ngrams == 7 :

		print('---------------------------------------\n----------------------------------------\n\ncharger le modéle 7 grams:  ... \n\n-----------------------------\n\n')
#traitement_Best_selon_test		
		
		# config = 'RF_max_depth : 38_criter : entropy_n_arbre_by_forest : 507grams'
				
		# chemin_model = "/home/medad/methode_par_apprentissage/resultats_apprentissage_ models Fevrier/RF/"
		# clf = load(chemin_model+config+'.joblib') 		
			
		# print('\n Resultats du teste du model RF en utilisant le corpus de validation\n')			
	
		# acc_test_corpus_test = predire(clf, X_validation, Y_validation ,config)

#traitement ALL
		chemin_model = "/home/medad/methode_par_apprentissage/resultats_apprentissage_ models Fevrier/RF/"
		import os
		for file in os.listdir(chemin_model):
			if file.endswith("7grams.joblib"):
				clf = load(chemin_model+file)
				acc_Validation = predire(clf, X_validation, Y_validation,file)



	
	# print('\n Resultats du teste du model SVM en utilisant le corpus d\'apprentissage \n')			
	
	# acc_corpus_train = predire(clf, X_train, Y_train,config)



def predire(model,X_test, Y_test,execution):
	
	print('#######################################  Résultat des PREDICTIONS sur le corpus de teste ################################################ \n')
	print('CONFIG : ' + execution)
	# calculate predictions
	predictions = model.predict(X_test)

	#print(predictions)





	
	# round predictions
	rounded = [round(x) for x in predictions]
	#rounded = [round(x[0]) for x in predictions]


	

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
	
	accuracy = (correct / i)*100

	print("le taux exactitude (nombre d'ESNN reconu sur le nombre total ESNN) : " + str(accuracy))
	return accuracy

#validation 227
#load_RF("/home/medad/methode_par_apprentissage/resultats_apprentissage_ models Fevrier/Validation_experimentation_juillet/Corpus_validation/Selon_sebastien/Corpus_validation_finale_227.csv", "/home/medad/methode_par_apprentissage/resultats_apprentissage_ models Fevrier/Validation_experimentation_juillet/Corpus_validation/Selon_sebastien/Y_Corpus_validation_finale_227.txt")
#validation 239
#load_RF("/home/medad/methode_par_apprentissage/resultats_apprentissage_ models Fevrier/Validation_experimentation_juillet/Corpus_validation/selon_Mauro/corpus_validation_sm_mg_shufle.csv", "/home/medad/methode_par_apprentissage/resultats_apprentissage_ models Fevrier/Validation_experimentation_juillet/Corpus_validation/selon_Mauro/Y_corpus_validation_sm_mg_shufle.txt")

#Emergence 93
#load_RF("/home/medad/methode_par_apprentissage/resultats_apprentissage_ models Fevrier/Validation_experimentation_juillet/copus_emergence/corpus_emergence_AM_ancien.csv", "/home/medad/methode_par_apprentissage/resultats_apprentissage_ models Fevrier/Validation_experimentation_juillet/copus_emergence/Y_corpus_emergence_AM_ancien.txt")


#corpus_test_mix
#load_RF("/home/medad/methode_par_apprentissage/resultats_apprentissage_ models Fevrier/Validation_experimentation_juillet/mix_corpus_test_validation/corpus_test_mix.csv", "/home/medad/methode_par_apprentissage/resultats_apprentissage_ models Fevrier/Validation_experimentation_juillet/mix_corpus_test_validation/Y_corpus_test_mix.txt")

#corpus_Validation_mix
load_RF("/home/medad/methode_par_apprentissage/resultats_apprentissage_ models Fevrier/Validation_experimentation_juillet/mix_corpus_test_validation/corpus_validation_mix.csv", "/home/medad/methode_par_apprentissage/resultats_apprentissage_ models Fevrier/Validation_experimentation_juillet/mix_corpus_test_validation/Y_corpus_validation_mix.txt")



#print ("\n\n*******************LES TESTES SUR LE CORPUS DE 764 CORIGE SANS ENSE******************\n")

#load_svm("../corpus/corpus_corigee/corpus_513/Corpus_general513_corrigee+negatifs_equilibre_sansESNE.csv", "../corpus/corpus_corigee/corpus_513/Y_general513_corrigee+negatifs_equilibre_sansESNE.txt")