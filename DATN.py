# Them cac thu vien

import numpy as np
import cv2
import imutils
from imutils import perspective
from imutils import contours
from scipy.spatial import distance as dist

# Khoi tao webcam
cap = cv2.VideoCapture(0)

# Khoi tao trung diem cho anh
def midPoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# Ham tim tieu cu may anh
def FocalLengthFinder(measuredDistance, realWidth, widthInRfImage):
    focalLength = (widthInRfImage * measuredDistance) / realWidth
    return focalLength
  
# Ham tim khoang cach tu Camera toi vat
def DistanceFinder(FocalLength, realWidth, widthInFrame):
    distance = (realWidth * FocalLength) / widthInFrame
    return distance

# Thiet lap tieu cu
focalStandard = 34.3

file = open("myData.csv", "a")


while (cap.read()):

        # Camera
        ref,frame = cap.read()
        lengthCamera = int(cap.get(3))
        widthCamera = int(cap.get(4))
        frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
        orig = frame[:lengthCamera,:widthCamera]


        # Chuyen anh mau sang anh xam
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Su dung bo loc nhieu GaussianBlur
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        # Chuyen anh xam sang anh nhi phan
        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
        kernel = np.ones((3,3),np.uint8)

        # Su dung module Opening de loc diem trang 
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)

        resultImg = opening.copy()
        contours,hierachy = cv2.findContours(resultImg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        countObject = 0

        pixelsPerMetric = None

        for cnt in contours:

            #Pembacaan Area Objek yang di Ukur
            area = cv2.contourArea(cnt)
            x,y,w,h = cv2.boundingRect(cnt)

            #Jika Area Kurang dari 1000 dan Lebih dari 12000  Pixel
            #Maka Lakukan Pengukuran
            if area < 1000 or area > 1200000:
                continue

            orig = frame.copy()
            box = cv2.minAreaRect(cnt)
            box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 64), 2)

            
            for (x, y) in box:
                cv2.circle(orig, (int(x), int(y)), 5, (0, 255, 64), -1)

            
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midPoint(tl, tr)
            (blbrX, blbrY) = midPoint(bl, br)
            (tlblX, tlblY) = midPoint(tl, bl)
            (trbrX, trbrY) = midPoint(tr, br)

 
            cv2.circle(orig, (int(tltrX), int(tltrY)), 0, (0, 255, 64), 5)
            cv2.circle(orig, (int(blbrX), int(blbrY)), 0, (0, 255, 64), 5)
            cv2.circle(orig, (int(tlblX), int(tlblY)), 0, (0, 255, 64), 5)
            cv2.circle(orig, (int(trbrX), int(trbrY)), 0, (0, 255, 64), 5)

 
            cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                    (255, 0, 255), 2)
            cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                    (255, 0, 255), 2)

  
            lengthPixel = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            widthPixel = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            # Doi don vi px sang cm
            if pixelsPerMetric is None:
                pixelsPerMetric = lengthPixel
                pixelsPerMetric = widthPixel
            
            length = lengthPixel
            width = widthPixel 

            widthConvert = round((width / focalStandard), 1)
            distanceMeasured = (focalStandard * widthConvert) / width
            focalLengthFound = FocalLengthFinder(distanceMeasured, widthConvert, width)
            distanceCamera = DistanceFinder(focalLengthFound, widthConvert, width)
            focalAutoChange = round(focalLengthFound/distanceCamera, 1)

            widthMeasured = round((width / focalAutoChange), 1)
            lengthMeasured = round((length / focalAutoChange), 1) 

            cv2.putText(orig, "L: {:.1f}CM".format(lengthMeasured),(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0,255), 2)
            cv2.putText(orig, "W: {:.1f}CM".format(widthMeasured),(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0,255), 2)
            countObject += 1

            file.write("\n" + "," + str(lengthMeasured) + "," + str(widthMeasured))

        cv2.putText(orig, "Object: {}".format(countObject),(10,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2, cv2.LINE_AA)  
        cv2.imshow('Camera',orig)    
        # print("W: ", widthMeasured, " cm\n")
        key = cv2.waitKey(1)
        if (key == 27):
            break

        
cap.release()
cv2.destroyAllWindows()





