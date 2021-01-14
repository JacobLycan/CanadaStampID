#A program to identify Canadian stamps by image
import cv2
import os

#Folder paths
templatePath =  'StampTemplate'
imagePath = "StampImages"
textPath = "StampInfo"
myList = os.listdir(templatePath) 

orb = cv2.ORB_create(nfeatures=1000)
images =[] #list of different stamps
classNames = [] #names of different stamps



def findDes(images):
    #Find the descriptors of a list of images
    desList =[]
    for img in images:
        kp,des= orb.detectAndCompute(img,None)
        desList.append(des)
    return desList





def findStamp(stamp,desList):
    #find the closest stamp to the provided image
    kp2,des2 = orb.detectAndCompute(stamp,None)
    bf = cv2.BFMatcher()
    matchList =[]
    finalVal =-1
    for des in desList:
        matches = bf.knnMatch(des,des2, k=2)

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        matchList.append(len(good))
    
    print (matchList)
    if len(matchList)!=0:
        finalVal = matchList.index(max(matchList))
    return finalVal



#For every stamp in the template folder find a descriptor
for cl in myList:
    imgCur = cv2.imread(f'{templatePath}/{cl}',0)
    images.append(imgCur)
    desList = findDes(images)
    classNames.append(os.path.splitext(cl)[0])


img = ""
while img!="Quit":
    #Get a user provided image and find a matching stamp 
    try:
        img = input("Enter a jpg file that is in the current directory ")
        img2 = cv2.imread(f'{imagePath}/{img}',0)
        origImg = cv2.imread(f'{imagePath}/{img}',-1)
        id = findStamp(img2,desList)
        if id!=-1:
            try:
                f = open(textPath+"/"+classNames[id]+".txt", "r") #Find the info about the stamp
                print(f.read())
            except:
                pass
            cv2.putText(origImg, classNames[id],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        cv2.imshow('img',origImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        pass





