import ML_assign_module
choice=0
while(True):
    print("1.LOGISTIC")
    print("2.DECISION")
    print("3.KNN")
    print("4.RandomForest")
    print("5.GRAPH")
    print("6.ExiT")
    choice=int(input("Enter your choice\n"))
    if(choice==1):
        lrobj=creditmodel.Logistic()
        choice=0
        while(True):
            print("1.TO TRAIN THE MACHINE")
            print("2.TO PREDICT THE VALUE")
            print("3.BACK")
            choice=int(input("Enter your choice\n"))
            if(choice==1):
                lrobj.fit1()
            elif(choice==2):
                lrobj.predict1()
            elif(choice==3):
                print("PREVIOUS MENU")
                break
    elif(choice==2):
        dtobj=creditmodel.Decision()
        choice=0
        while(True):
            print("1.TO TRAIN THE MACHINE")
            print("2.TO PREDICT THE VALUE")
            print("3.BACK")
            choice=int(input("Enter your choice\n"))
            if(choice==1):
                dtobj.fit1()
            elif(choice==2):
                dtobj.predict1()
            elif(choice==3):
                print("PREVIOUS MENU")
                break
    elif(choice==3):
        knobj=creditmodel.Knn()
        choice=0
        while(True):
            print("1.TO TRAIN THE MACHINE")
            print("2.TO PREDICT THE VALUE")
            print("3.BACK")
            choice=int(input("Enter your choice\n"))
            if(choice==1):
                knobj.fit1()
            elif(choice==2):
                knobj.predict1()
            elif(choice==3):
                print("PREVIOUS MENU")
                break
    elif(choice==4):
        rfobj=creditmodel.RandomForest()
        choice=0
        while(True):
            print("1.TO TRAIN THE MACHINE")
            print("2.TO PREDICT THE VALUE")
            print("3.BACK")
            choice=int(input("Enter your choice\n"))
            if(choice==1):
                rfobj.fit1()
            elif(choice==2):
                rfobj.predict1()
            elif(choice==3):
                print("PREVIOUS MENU")
                break

    elif(choice==5):
        gobj=creditmodel.Graph()
        gobj.visualization()
    elif(choice==6):
        print("EXITED")
        break
    
