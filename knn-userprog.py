import knn_module
obj1=module.Diabetes()
choice=0
while(True):
    print("1.TO TRAIN THE MACHINE")
    print("2.TO PREDICT THE VALUE")
    print("3.TO GIVE VALUES MANUALLY")
    choice=int(input("Enter your choice\n"))
    if(choice==1):
        obj1.fit()
    elif(choice==2):
        obj1.predict()
    elif(choice==3):
        obj1.manvalue()
    elif(choice==4):
        break
    
