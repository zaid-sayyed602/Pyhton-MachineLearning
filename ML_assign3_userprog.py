import ML_assign3_module
knobj=module.Knn()
choice=0
while(True):
    print("1.SVM")
    print("2.DECISION")
    print("3.KNN")
    choice=int(input("Enter your choice\n"))
    if(choice==1):
        choice=0
        while(True):
            print("1.TO TRAIN THE MACHINE")
            print("2.TO PREDICT THE VALUE")
            print("3.BACK")
            choice=int(input("Enter your choice\n"))
            if(choice==1):
                knobj.fit()
            elif(choice==2):
                knobj.predict()
            elif(choice==3):
                print("PREVIOUS MENU")
                break
    elif(choice==2):
        choice=0
        while(True):
            print("1.TO TRAIN THE MACHINE")
            print("2.TO PREDICT THE VALUE")
            print("3.BACK")
            choice=int(input("Enter your choice\n"))
            if(choice==1):
                drobj.fit()
            elif(choice==2):
                drobj.predict()
            elif(choice==3):
                print("PREVIOUS MENU")
                break
    elif(choice==3):
        choice=0
        while(True):
            print("1.TO TRAIN THE MACHINE")
            print("2.TO PREDICT THE VALUE")
            print("3.TO GIVE MANUAL VALUES")
            print("4.GRAPH")
            print("5.BACK")
            choice=int(input("Enter your choice\n"))
            if(choice==1):
                knobj.fit()
            elif(choice==2):
                knobj.predict()
            elif(choice==3):
                knobj.manvalue()
            elif(choice==4):
                knobj.graph()
            elif(choice==5):
                print("PREVIOUS MENU")
                break
    elif(choice==5):
        print("Exited")
        break
    
