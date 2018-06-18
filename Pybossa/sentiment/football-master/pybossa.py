#!/usr/bin/env python2

from __future__ import division
import json
import codecs
import pandas as pd
import numpy as np
import unicodedata
from kappa import fleiss_kappa


def tweetsForPybossa(jsonfile):
##CELDA 1
    #En esta celda extraemos del json del streaming el tweet e id_str y lo guardamos en un csv llamado
    #tweets.csv en la carpeta csv
    json_data= codecs.open(jsonfile,"r", "utf-8")
    ############################################################################
    arraytowrite = []

    for renglon in json_data:
        data = json.loads(renglon)
        arraytowrite.append(data["text"])

    arraytowrite = list(set(arraytowrite))

    arraytowrite = pd.DataFrame(data=arraytowrite, columns=["Tweet"])

    #Se pueden poner los indices del array o no
    arraytowrite.to_csv('./csv/tweets.csv', columns=["Tweet"], index=False, encoding="utf-8")

    json_data.close()

##CELDA 2

    # CREATING A DICTIONARY OF GOLDEN ANSWERS WITH THE STRUCTURE TWEETGOLDEN : LABEL
    golden = pd.read_csv('./csv/Golden.csv', encoding="utf-8")
    tweetsgold = golden['Tweet']
    labelgold = golden['Etiqueta']

    dictgolden = {}
    indexgolden = 0
    for i in tweetsgold:
        dictgolden[i] = labelgold[indexgolden]
        indexgolden += 1

    ##ADDING GOLDEN ANSWERS TO TWEETS CSV
    tweets = pd.read_csv('./csv/tweets.csv', encoding="utf-8")
    tweets = tweets['Tweet']
    n = int(len(set(tweets)) / len(tweetsgold))

    indextweets = 0
    indexgolden = 0
    listcsv = []

    for i in tweets:
        if (i not in listcsv):
            listcsv.append(i)
            if((indextweets%n == 0) and (indexgolden<len(tweetsgold))):
                listcsv.append(tweetsgold[indexgolden])
                indexgolden += 1
        indextweets += 1


    listcsv = pd.DataFrame(data=listcsv, columns=["Tweet"])
    listcsv.to_csv('tweetsForPybossa.csv', index=False, encoding="utf-8")


def pybossaReport(numberofcategories):
    # SAVING PROJECT NAME
    project_name =""
    with open('project.json') as json_data:
        data = json.load(json_data)
        project_name = data['short_name']
        json_data.close()

    df = pd.read_json(project_name + '_task.json', encoding="utf-8")


    golden = pd.read_csv('./csv/Golden.csv', encoding="utf-8")
    tweetsgold = golden['Tweet']
    labelgold = golden['Etiqueta']

    dictgolden = {}
    indexgolden = 0
    for i in tweetsgold:
        dictgolden[i] = labelgold[indexgolden]
        indexgolden += 1


    dictidtweet = {}


    ########## CREAMOS EL DICCIONARIO ID : TWEET

    index = 0
    for d in df["id"]:

        dictidtweet[d] = df["info"][index]["Tweet"]

        #dictidtweet[d["id"]] = 0
        index = index + 1

    #print(dictidtweet)

    #Diccionario del task-run  {task_id : info}
    ########## CARGAMOS EL JSON CON LOS ID Y LAS RESPUESTAS QUE SACAMOS DE PYBOSSA

    dfrun = pd.read_json(project_name + '_task_run.json', encoding="utf-8")


    dictidinfo = {}

    dicttasks = {}

    numtagr = []

    ##PARA HALLAR LA CONFIANZA EN UN USUARIO
    usergolden = []
    useragree = []

    userslist = []
    infoslist = []
    answerslistwithtweet=["Tweet"]
    answerslist = []
    tweetslist = []
    twlist = []

    indexrun = 0

    for d in dfrun["task_id"]:
        usuario = 0

        ##HACER LAS OPERACIONES PARA DEFINIR LA ETIQUETA CONCRETA##
        tweetslist.append(dictidtweet[d])

    
        usuario = dfrun["user_id"][indexrun]

        userslist.append(usuario)
        info = dfrun["info"][indexrun]
        infoslist.append(info)
    
        if (info not in answerslist):
            answerslist.append(info)
            answerslistwithtweet.append(info)
    
        if (dictidtweet[d] not in twlist):
            twlist.append(dictidtweet[d])

        
        indexrun += 1

    print("LONG TWEETLIST")
    print(len(tweetslist))

    print("LONG TWLIST")
    print(len(twlist))

    print("ANSWERS LISTS")
    print(answerslist)

    ########## ASOCIAMOS EN UN DATAFRAME EL USUARIO, EL TWEET Y SU RESPUESTA

    arraytowrite = np.column_stack((userslist, tweetslist, infoslist))

    arraytowrite = pd.DataFrame(data=arraytowrite, columns=["User","Tweet","Info"])

    #Se pueden poner los indices del array o no
    arraytowrite.to_csv('./csv/usertweetinfo.csv', index=False, encoding="utf-8")



    ##QUEDA ASOCIAR CADA PREGUNTA A UN NUMERO PARA HACER KAPPA
        
    #print(dicttasks)

    print(project_name)
    indexrunprueba = 0
    for d in dfrun["task_id"]:

        dictidinfo[d] = dfrun["info"][indexrunprueba]

        #dictidtweet[d["id"]] = 0
        indexrunprueba = indexrunprueba + 1


    print(dictidinfo)

    ##### PROGRAMAR LO DEL ENKI #####

    dict_top_answers = {}
    dictinfos = {}
    checked = []

    for d in dfrun["task_id"]:
        if (d not in checked):
            for info in infoslist:
                dictinfos[info] = 0

            print(dictinfos)

            index = 0

            for e in dfrun["task_id"]:
                if (d == e):
                    info = dfrun["info"][index]
                    dictinfos[info] +=1 
            
                else:
                    index += 1

            result = ""
            top = 0
            for inf in infoslist:
                if (dictinfos[inf] > top):
                    top = dictinfos[inf]
                    result = inf

            dict_top_answers[d] = result
            checked.append(d)


    print("DICCIONARIO TOP ANSWERS")
    print(dict_top_answers)


    listatweets = []
    listainfos = []

    for d in dict_top_answers:
    	#listatweets.append(d.keys())
    	listainfos.append(d)


    for l in listainfos:
    	listatweets.append(dict_top_answers[l])

    print("LISTA infos")
    print(listainfos)

    print("LISTA TWEETS")
    print(listatweets)

    dict_top_answers2 = np.column_stack((listatweets, listainfos))

    dict_top_answers2 = pd.DataFrame(data=dict_top_answers2, columns=["Tweet","Info"])

    dict_top_answers2.to_csv('./csv/dict_top_answers.csv', index=False, encoding="utf-8")
    #dict_top_answers2 = pd.DataFrame(data=dict_top_answers, columns=["Tweet","Info"])


    #dict_top_answers.to_csv('./csv/dict_top_answers.csv')


    ########## EXTRAEMOS PARA CADA USUARIO LAS RESPUESTAS A LAS PREGUNTAS GOLDEN PARA COMPARARLAS CON SUS RESPUESTAS CORRECTAS 

    puntuacionesgolden={}
    puntuacionesmayoria={}

    us = set(userslist)
    for i in us:
        puntuacionesgolden[i]=0
        puntuacionesmayoria[i]=0
        
    index = 0
    uti = pd.read_csv('./csv/usertweetinfo.csv', encoding="utf-8")

    for i in uti["Tweet"]:
        if i in list(dictgolden.keys()):
            if (uti["Info"][index] == dictgolden[i]):
                puntuacionesgolden[uti["User"][index]] += 1
        index +=1


    for i in us:
        puntuacionesgolden[i] = float(puntuacionesgolden[i] / len(dictgolden))

    print("PUNTUACIONES GOLDEN")
    print(puntuacionesgolden)


    ########## EXTRAEMOS PARA CADA USUARIO LAS RESPUESTAS A TODAS LAS PREGUNTAS PARA COMPARARLAS CON LAS RESPUESTAS 

    tweetinfotop = {}

    for i in list(dict_top_answers.keys()):
        tweetinfotop[dictidtweet[i]] = dict_top_answers[i]    

    index = 0

    for i in uti["Tweet"]:
        if i not in list(dictgolden.keys()):
            if (uti["Info"][index] == tweetinfotop[i]):
                puntuacionesmayoria[uti["User"][index]] += 1
        index +=1


    for i in us:
        ### HEMOS CAMBIADO ESTO
        puntuacionesmayoria[i] = float(puntuacionesmayoria[i] / (len(twlist) - len(dictgolden) ))
        
    print(puntuacionesmayoria)


    goldentrust = []
    majoritytrust = []

    index = 0
    for i in uti["Tweet"]:
        usuario = uti["User"][index]
        goldentrust.append(puntuacionesgolden[usuario])
        majoritytrust.append(puntuacionesmayoria[usuario])
        index += 1
        
    goldentrust = pd.DataFrame(goldentrust)
    majoritytrust = pd.DataFrame(majoritytrust)

    uti['Golden trust'] = goldentrust
    uti['Majority trust'] = majoritytrust
    uti.to_csv('./csv/usertweetinfo.csv', encoding='utf-8', index = False)

    # Hay que hallar la suma de las confianzas para cada pregunta
    # y la suma de confianzas para cada pregunta y respuesta.

    userslist = list(set(userslist))
    listgolden = list(dictgolden.keys())

    sumgolden = 0
    summayor = 0

    for i in userslist:
        sumgolden += puntuacionesgolden[i]
        summayor += puntuacionesmayoria[i]
        
    print(sumgolden)
    print(summayor)


    answerstdgolden = {"Tweet":[]}
    answerstdmaj = {"Tweet":[]}

    for i in answerslist:
        answerstdgolden[i] = []
        answerstdmaj[i] = []


    answersdictg = {}
    answersdictm = {}    
        
    for i in list(dictidtweet.keys()):
        index = 0
        for z in answerslist:
            answersdictg[z] = 0
            answersdictm[z] = 0    
        #if i in listgolden:
        tw = dictidtweet[i]
        
        if (tw not in listgolden):
            for j in uti['Tweet']:
                if (tw == j):
                    twit = j
                    for k in answerslistwithtweet:
                        if (k == uti['Info'][index]):
                            answersdictg[k] += uti['Golden trust'][index]
                            answersdictm[k] += uti['Majority trust'][index]
                index += 1
            
            aux = answerstdgolden['Tweet']
            aux.append(twit)
            answerstdgolden['Tweet'] = aux
            answerstdmaj['Tweet'] = aux
                #Dividir entre sumgolden y summayor
            for l in list(answersdictg.keys()):
                resultgold = float(answersdictg[l] / sumgolden)
                resultmaj = float(answersdictm[l] / summayor)
                auxgold = answerstdgolden[l]
                auxmaj = answerstdmaj[l]
                auxgold.append(resultgold)
                auxmaj.append(resultmaj)
                answerstdgolden[l] = auxgold
                answerstdmaj[l] = auxmaj



    percentagesgolden = pd.DataFrame(data=answerstdgolden, columns=answerslistwithtweet)
    percentagesmaj = pd.DataFrame(data=answerstdmaj, columns=answerslistwithtweet)


    percentagesmaj.to_csv('./csv/majoritypercent.csv', index=False, encoding="utf-8")
    percentagesgolden.to_csv('./csv/goldenpercent.csv', index=False, encoding="utf-8")


    ## KAPPAAAAAAAAAAAAAA

    index2 = 0

    
    answersdict = answersdictg
    pos = 0
    for i in answersdict.keys():
        answersdict[i] = pos
        pos = pos + 1
    
 


    allex = []
    

    for i in list(dictidtweet.keys()):
        counter = 0
        exnewkappa = []
        j = 0
        while (j<numberofcategories):
            exnewkappa.append(0)
            j += 1

        tweet = dictidtweet[i]
        index = 0

        for l in uti["Tweet"]:

            if (l == tweet):
                print(l)
                for a in answerslist:
                    if (a == uti["Info"][index]):
                        #ex.append( (index2,answerslist.index(a)) )
                        arrayindex = answersdict[a]
                        counter += 1
                        exnewkappa[arrayindex] += 1

                if(counter == len(userslist)):
                    #pdb.set_trace()
        
                    allex.append(exnewkappa)
                  


            index += 1
    
    allex.append(exnewkappa)

    print(allex)

    fk = fleiss_kappa(allex)

 
    print(fk)