




def get_data():
    filename = "data/HU_sentences_results.txt"
    humanamount = 0
    humansum = 0
    humanlist = []
    maxx = 50
    with open(filename, "r") as fl:
        for sentence in fl.readlines():                         ############# RESULTS #############
            liste =[]                                           # AI average is 0.23820649143351932
            x = sentence.find("[")                              # Human average is 0.09411602139188101
            y = sentence.find("]")
            new = sentence[x+1:y]

            for item in new.split(","):
                humansum += float(item)
                humanamount += 1
                liste.append(float(item))

            humanlist.append(liste)
    filename = "data/AI_sentences_results.txt"
    aiamount = 0
    aisum = 0
    ailist = []  #here you can later append all the data to. Ideally keep it a list of lists
    with open(filename, "r") as fl:
        for sentence in fl.readlines():
            liste =[]
            x = sentence.find("[")
            y = sentence.find("]")
            new = sentence[x+1:y]

            for item in new.split(","):
                aisum += float(item)
                aiamount += 1
                liste.append(float(item))

            ailist.append(liste)

    for liste in ailist:
        länge = len(liste)
        for i in range(maxx-länge):

            liste.append(0.0)
        assert len(liste) == 50

    for liste in humanlist:
        länge = len(liste)
        for i in range(maxx-länge):
            liste.append(0.0)

        if len(liste) > 50:
            print(len(liste))



    return humanlist, ailist
#   print("this is the ai list ")
#   print(ailist)


get_data()