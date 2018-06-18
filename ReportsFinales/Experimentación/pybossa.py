#!/usr/bin/env python2

from __future__ import division
import json
import codecs
import pandas as pd
import numpy as np
import unicodedata
from kappa import fleiss_kappa


def tweetsForPybossaJson(jsonwithidspath, goldencsvpath):

    if ((goldencsvpath) != None):
    ##CELDA 1
        #En esta celda extraemos del json del streaming el tweet e id_str y lo guardamos en un csv llamado
        #tweets.csv en la carpeta csv
        json_data= codecs.open(jsonwithidspath,"r", "utf-8")
        ############################################################################
        arraytowrite = []

        for renglon in json_data:
            data = json.loads(renglon)
            arraytowrite.append(data["text"])

        arraytowrite = list(set(arraytowrite))

        arraytowrite = pd.DataFrame(data=arraytowrite, columns=["Tweet"])


        json_data.close()

    ##CELDA 2

        # CREATING A DICTIONARY OF GOLDEN ANSWERS WITH THE STRUCTURE TWEETGOLDEN : LABEL
        golden = pd.read_csv(goldencsvpath, encoding="utf-8")
        tweetsgold = golden['Tweet']
        labelgold = golden['Etiqueta']

        dictgolden = {}
        indexgolden = 0
        for i in tweetsgold:
            dictgolden[i] = labelgold[indexgolden]
            indexgolden += 1

        ##ADDING GOLDEN ANSWERS TO TWEETS CSV

        tweets = arraytowrite['Tweet']
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



    else:
##CELDA 1
    #En esta celda extraemos del json del streaming el tweet e id_str y lo guardamos en un csv llamado
    #tweets.csv en la carpeta csv
        json_data= codecs.open(jsonwithidspath,"r", "utf-8")
        ############################################################################
        arraytowrite = []

        for renglon in json_data:
            data = json.loads(renglon)
            arraytowrite.append(data["text"])

        arraytowrite = list(set(arraytowrite))

        arraytowrite = pd.DataFrame(data=arraytowrite, columns=["Tweet"])


        json_data.close()


        tweets = arraytowrite['Tweet']

        indextweets = 0
        listcsv = []

        for i in tweets:
            if (i not in listcsv):
                listcsv.append(i)

        listcsv = pd.DataFrame(data=listcsv, columns=["Tweet"])
        listcsv.to_csv('tweetsForPybossa.csv', index=False, encoding="utf-8")



def tweetsForPybossaCsv(csvfile, goldencsvpath):
    
    if(goldencsvpath != None):
        golden = pd.read_csv(goldencsvpath, encoding="utf-8")
        tweetsgold = golden['Tweet']
        labelgold = golden['Etiqueta']

        dictgolden = {}
        indexgolden = 0
        for i in tweetsgold:
            dictgolden[i] = labelgold[indexgolden]
            indexgolden += 1


        tweets = pd.read_csv(csvfile, encoding="utf-8")
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


    else:
        tweets = pd.read_csv(csvfile, encoding="utf-8")
        tweets = tweets['Tweet']

        indextweets = 0
        indexgolden = 0
        listcsv = []

        for i in tweets:
            if (i not in listcsv):
                listcsv.append(i)


        listcsv = pd.DataFrame(data=listcsv, columns=["Tweet"])
        listcsv.to_csv('tweetsForPybossa.csv', index=False, encoding="utf-8")


def pybossaReport(filewithids, taskjsonpath, taskrunjsonpath, numberofcategories, goldencsvpath):
    
    if(goldencsvpath != None):
        idstr = pd.read_csv(filewithids, encoding = 'utf-8')
        idstrdict = {}
        index = 0

        for i in idstr["Tweet"]:
            idstrdict[i] = idstr["id_str"][index]
            index += 1

        df = pd.read_json(taskjsonpath, encoding="utf-8")


        golden = pd.read_csv(goldencsvpath, encoding="utf-8")
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

        dfrun = pd.read_json(taskrunjsonpath, encoding="utf-8")


        dictidinfo = {}

        dicttasks = {}

        numtagr = []

        ##PARA HALLAR LA CONFIANZA EN UN USUARIO
        usergolden = []
        useragree = []

        userslist = []
        infoslist = []
        answerslistwithtweet=["id_str","Tweet"]
        answerslist = []
        tweetslist = []
        twlist = []
        identi = []

        indexrun = 0

        for d in dfrun["task_id"]:
            usuario = 0

            tweetslist.append(dictidtweet[d])

            if(dictidtweet[d] not in dictgolden.keys()):
                identi.append(idstrdict[dictidtweet[d]])

            else:
                identi.append("Golden question")
        
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


        ########## ASOCIAMOS EN UN DATAFRAME EL USUARIO, EL TWEET Y SU RESPUESTA

        arraytowrite = np.column_stack((userslist, identi, tweetslist, infoslist))

        arraytowrite = pd.DataFrame(data=arraytowrite, columns=["User","id_str","Tweet","Info"])

        #Se pueden poner los indices del array o no
        arraytowrite.to_csv('usertweetinfo.csv', index=False, encoding="utf-8")



        ##QUEDA ASOCIAR CADA PREGUNTA A UN NUMERO PARA HACER KAPPA
            
        #print(dicttasks)

        indexrunprueba = 0
        for d in dfrun["task_id"]:

            dictidinfo[d] = dfrun["info"][indexrunprueba]

            #dictidtweet[d["id"]] = 0
            indexrunprueba = indexrunprueba + 1



        ##### PROGRAMAR LO DEL ENKI #####

        dict_top_answers = {}
        dictinfos = {}
        checked = []

        for d in dfrun["task_id"]:
            if (d not in checked):
                for info in infoslist:
                    dictinfos[info] = 0


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




        ########## EXTRAEMOS PARA CADA USUARIO LAS RESPUESTAS A LAS PREGUNTAS GOLDEN PARA COMPARARLAS CON SUS RESPUESTAS CORRECTAS 

        puntuacionesgolden={}
        puntuacionesmayoria={}

        us = set(userslist)
        for i in us:
            puntuacionesgolden[i]=0
            puntuacionesmayoria[i]=0
            
        index = 0
        uti = pd.read_csv('usertweetinfo.csv', encoding="utf-8")

        for i in uti["Tweet"]:
            if i in list(dictgolden.keys()):
                if (uti["Info"][index] == dictgolden[i]):
                    puntuacionesgolden[uti["User"][index]] += 1
            index +=1


        for i in us:
            puntuacionesgolden[i] = float(puntuacionesgolden[i] / len(dictgolden))



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
        uti.to_csv('usertweetinfo.csv', encoding='utf-8', index = False)

        # Hay que hallar la suma de las confianzas para cada pregunta
        # y la suma de confianzas para cada pregunta y respuesta.

        userslist = list(set(userslist))
        listgolden = list(dictgolden.keys())

        sumgolden = 0
        summayor = 0

        for i in userslist:
            sumgolden += puntuacionesgolden[i]
            summayor += puntuacionesmayoria[i]
            


        answerstdgolden = {"id_str":[], "Tweet":[]}
        answerstdmaj = {"id_str":[], "Tweet":[]}

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
                
                ids = answerstdgolden['id_str']
                aux = answerstdgolden['Tweet']
                aux.append(twit)
                ids.append(idstrdict[twit])

                answerstdgolden['id_str'] = ids
                answerstdmaj['id_str'] = ids
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


        percentagesmaj.to_csv('majoritypercent.csv', index=False, encoding="utf-8")
        percentagesgolden.to_csv('goldenpercent.csv', index=False, encoding="utf-8")


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


        fk = fleiss_kappa(allex)

     
        print(fk)



    else:    
        idstr = pd.read_csv(filewithids, encoding = 'utf-8')
        idstrdict = {}
        index = 0

        for i in idstr["Tweet"]:
            idstrdict[i] = idstr["id_str"][index]
            index += 1

        df = pd.read_json(taskjsonpath, encoding="utf-8")

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

        dfrun = pd.read_json(taskrunjsonpath, encoding="utf-8")


        dictidinfo = {}

        dicttasks = {}

        numtagr = []

        ##PARA HALLAR LA CONFIANZA EN UN USUARIO
        useragree = []

        userslist = []
        infoslist = []
        answerslistwithtweet=["id_str","Tweet"]
        answerslist = []
        tweetslist = []
        twlist = []
        identi = []

        indexrun = 0

        for d in dfrun["task_id"]:
            usuario = 0

            tweetslist.append(dictidtweet[d])

            
            identi.append(idstrdict[dictidtweet[d]])
        
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


        ########## ASOCIAMOS EN UN DATAFRAME EL USUARIO, EL TWEET Y SU RESPUESTA

        arraytowrite = np.column_stack((userslist, identi, tweetslist, infoslist))

        arraytowrite = pd.DataFrame(data=arraytowrite, columns=["User","id_str","Tweet","Info"])

        #Se pueden poner los indices del array o no
        arraytowrite.to_csv('usertweetinfo.csv', index=False, encoding="utf-8")



        ##QUEDA ASOCIAR CADA PREGUNTA A UN NUMERO PARA HACER KAPPA
            
        #print(dicttasks)

        indexrunprueba = 0
        for d in dfrun["task_id"]:

            dictidinfo[d] = dfrun["info"][indexrunprueba]

            #dictidtweet[d["id"]] = 0
            indexrunprueba = indexrunprueba + 1



        ##### PROGRAMAR LO DEL ENKI #####

        dict_top_answers = {}
        dictinfos = {}
        checked = []

        for d in dfrun["task_id"]:
            if (d not in checked):
                for info in infoslist:
                    dictinfos[info] = 0


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




        ########## EXTRAEMOS PARA CADA USUARIO LAS RESPUESTAS A LAS PREGUNTAS GOLDEN PARA COMPARARLAS CON SUS RESPUESTAS CORRECTAS 

        puntuacionesmayoria={}

        us = set(userslist)
        for i in us:
            puntuacionesmayoria[i]=0
            
        index = 0
        uti = pd.read_csv('usertweetinfo.csv', encoding="utf-8")

        ########## EXTRAEMOS PARA CADA USUARIO LAS RESPUESTAS A TODAS LAS PREGUNTAS PARA COMPARARLAS CON LAS RESPUESTAS 

        tweetinfotop = {}

        for i in list(dict_top_answers.keys()):
            tweetinfotop[dictidtweet[i]] = dict_top_answers[i]    

        index = 0

        for i in uti["Tweet"]:
            if (uti["Info"][index] == tweetinfotop[i]):
                puntuacionesmayoria[uti["User"][index]] += 1
            index +=1


        for i in us:
            ### HEMOS CAMBIADO ESTO
            puntuacionesmayoria[i] = float(puntuacionesmayoria[i] / (len(twlist) ))
            

        majoritytrust = []

        index = 0
        for i in uti["Tweet"]:
            usuario = uti["User"][index]
            majoritytrust.append(puntuacionesmayoria[usuario])
            index += 1
            
        majoritytrust = pd.DataFrame(majoritytrust)

        uti['Majority trust'] = majoritytrust
        uti.to_csv('usertweetinfo.csv', encoding='utf-8', index = False)

        # Hay que hallar la suma de las confianzas para cada pregunta
        # y la suma de confianzas para cada pregunta y respuesta.

        userslist = list(set(userslist))

        summayor = 0

        for i in userslist:
            summayor += puntuacionesmayoria[i]
            

        answerstdmaj = {"id_str":[], "Tweet":[]}

        for i in answerslist:
            answerstdmaj[i] = []


        answersdictm = {}

        for i in list(dictidtweet.keys()):
            index = 0
            for z in answerslist:
                answersdictm[z] = 0    
            #if i in listgolden:
            tw = dictidtweet[i]
            
            for j in uti['Tweet']:
                if (tw == j):
                    twit = j
                    for k in answerslistwithtweet:
                        if (k == uti['Info'][index]):
                            answersdictm[k] += uti['Majority trust'][index]
                index += 1
            
            ids = answerstdmaj['id_str']
            aux = answerstdmaj['Tweet']
            aux.append(twit)
            ids.append(idstrdict[twit])

            answerstdmaj['id_str'] = ids
            answerstdmaj['Tweet'] = aux
                #Dividir entre sumgolden y summayor
            for l in list(answersdictm.keys()):
                resultmaj = float(answersdictm[l] / summayor)
                auxmaj = answerstdmaj[l]
                auxmaj.append(resultmaj)
                answerstdmaj[l] = auxmaj


        percentagesmaj = pd.DataFrame(data=answerstdmaj, columns=answerslistwithtweet)


        percentagesmaj.to_csv('majoritypercent.csv', index=False, encoding="utf-8")


        ## KAPPAAAAAAAAAAAAAA

        index2 = 0

        
        answersdict = answersdictm
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


        fk = fleiss_kappa(allex)

     
        print(fk)