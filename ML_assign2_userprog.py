import ML_assign2_module
lrobj=module.LogisticRegression()
drobj=module.DecisionTree()
knobj=module.Knn()
gpobj=module.Graph()
choice=0
while(True):
    print("1.LOGISTIC")
    print("2.DECISION")
    print("3.KNN")
    print("4.Graph")
    choice=int(input("Enter your choice\n"))
    if(choice==1):
        choice=0
        while(True):
            print("1.TO TRAIN THE MACHINE")
            print("2.TO PREDICT THE VALUE")
            print("3.BACK")
            choice=int(input("Enter your choice\n"))
            if(choice==1):
                lrobj.fit()
            elif(choice==2):
                lrobj.predict()
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
            print("3.BACK")
            choice=int(input("Enter your choice\n"))
            if(choice==1):
                knobj.fit()
            elif(choice==2):
                knobj.predict()
            elif(choice==3):
                print("PREVIOUS MENU")
                break
    elif(choice==4):
        gpobj.bar()
    elif(choice==5):
        print("Exited")
        break
    
