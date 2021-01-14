import cv2
import os 


templatePath =  'StampTemplate'
imagePath = "StampImages"
textPath = "StampInfo"
orb = cv2.ORB_create(nfeatures=1000)
images =[]
classNames = []
myList = os.listdir(templatePath)
def findDes(images):
    desList =[]
    for img in images:
        kp,des= orb.detectAndCompute(img,None)
        desList.append(des)
    return desList





def findID(img,desList):
    kp2,des2 = orb.detectAndCompute(img,None)
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
print(myList)
for cl in myList:
    imgCur = cv2.imread(f'{templatePath}/{cl}',0)
    images.append(imgCur)
    desList = findDes(images)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

img = "hi"
while img!="Quit":
    try:
        img = input("Enter a jpg file that is in the current directory ")
        img2 = cv2.imread(f'{imagePath}/{img}',0)
        origImg = cv2.imread(f'{imagePath}/{img}',-1)
        id = findID(img2,desList)
        if id!=-1:
            try:
                f = open(textPath+"/"+classNames[id]+".txt", "r")
                print(f.read())
            except:
                pass
            cv2.putText(origImg, classNames[id],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        cv2.imshow('img',origImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        pass





