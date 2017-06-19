import cv2
import numpy as np
import sys
import time
import math
import random as rn
import copy
import sudokuS
import datetime

cap = cv2.VideoCapture(0)

class puzzleStatusClass:
    #this class defines the actual mathematical properties of the sudoku puzzle
    #and the associated methods used to solve the actual sudoku puzzle
    def __init__(self):
        #.SDKarray is a 9x9 grid of all solved values for the puzzle
        self.last = np.zeros((9,9),np.uint8)
        self.SDKarray = np.zeros((9,9),np.uint8) 
        self.solvedSDKarray = np.zeros((9,9),np.uint8)
        self.SDKdict = {}
        self.solvedSDKdict = {}
	self.check = None
	self.hintCell = 0

    def getSDKString(self):
        SDKString = ''
	for i in range(9):
	    for j in range(9):
	        SDKString = SDKString + str(self.SDKarray[i][j])
	return SDKString

    def arrToDict(self,y,x):
	row = ''
	if y == 0:
	    row = 'A'
	elif y == 1:
	    row = 'B'
	elif y == 2:
	    row = 'C'
	elif y == 3:
	    row = 'D'	
	elif y == 4:
	    row = 'E'
	elif y == 5:
	    row = 'F'
	elif y == 6:
	    row = 'G'
	elif y == 7:
	    row = 'H'
	elif y == 8:
	    row = 'I'

	column = str(x+1)
	
	return row+column

    def dictToArr(self,cell):
	y = 0
	x = 0 
	row = cell[0]
	if row == 'A':
	    y = 0
	elif row == 'B':
	    y = 1
	elif row == 'C':
	    y = 2
	elif row == 'D':
	    y = 3
	elif row == 'E':
	    y = 4
	elif row == 'F':
	    y = 5
	elif row == 'G':
	    y = 6
	elif row == 'H':
	    y = 7
	elif row == 'I':
	    y = 8
	x = int(cell[1])-1

	return y,x

	    
    def getSDKdict(self):
	Dict = {}
	for i in range(9):
	    for j in range(9):
		Dict[self.arrToDict(i,j)] = self.SDKarray[i][j]
	self.SDKdict = Dict

    def solve(self):
	self.solvedSDKdict = sudokuS.solve(self.getSDKString())
	if self.solvedSDKdict != False:
		for i in range(9):
		    for j in range(9):
			self.solvedSDKarray[i][j] = self.solvedSDKdict[self.arrToDict(i,j)]	



    def isValid(self):
	for k1 in self.SDKdict.keys():
	    for k2 in sudokuS.peers[k1]:
		if (self.SDKdict[k1] != 0) and (self.SDKdict[k1] == self.SDKdict[k2]):
		    return k1,k2	
	return True
		

    def checkSDK(self):
	iCheck = self.isValid()
	if iCheck == True:
	    self.solve()
	    if self.solvedSDKdict == False:
		self.check = 0
	    else:
		self.check = 1
	else:
	    self.check = iCheck
	    
    def getHintCell(self):
	self.hintCell = 0
	for k in self.SDKdict.keys():
	    if self.SDKdict[k] == 0:
	        self.hintCell = k		       
		break
	self.hintCell = self.dictToArr(self.hintCell)

class imageClass:
    #this class defines all of the important image matrices, and information about the images.
    #also the methods associated with capturing input, displaying the output,
    #and warping and transforming any of the images to assist with OCD
    def __init__(self):
        #.captured is the initially captured image
        self.captured = None
        #.gray is the grayscale captured image
        self.gray = []
        #.thres is after adaptive thresholding is applied
        self.thresh = []
        #.contours contains information about the contours found in the image
        self.contours = []
        #.biggest contains a set of four coordinate points describing the
        #contours of the biggest rectangular contour found
        self.biggest = None;
        #.maxArea is the area of this biggest rectangular found
        self.maxArea = 0
        #.output is an image resulting from the warp() method
        self.output = []
        self.outputBackup = []
        self.outputGray = []
        #.mat is a matrix of 100 points found using a simple gridding algorithm
        #based on the four corner points from .biggest
        self.mat = np.zeros((100,2),np.float32)
        #.reshape is a reshaping of .mat
        self.reshape = np.zeros((100,2),np.float32)
        
    def captureImage(self,status):
        #captures the image and finds the biggest rectangle
        try:
	    #rgb,_ = freenect.sync_get_video()
	    ret, rgb = cap.read()
           # bgr = cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)
            self.captured = rgb
        except TypeError:
            print "No Kinect Detected!"
            print "Loading sudoku.jpg..."
            #for testing purposes
            img = cv2.imread('2.png')
            self.captured = cv2.resize(img,(600,600))

        #convert to grayscale
        self.gray = cv2.cvtColor(self.captured, cv2.COLOR_BGR2GRAY)

        #noise removal with gaussian blur
        self.gray = cv2.GaussianBlur(self.gray,(5,5),0)
        #then do adaptive thresholding
        self.thresh = cv2.adaptiveThreshold(self.gray,255,1,1,11,2)

        #find countours in threshold image
        self.contours, hierarchy = cv2.findContours(self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #evaluate all blobs to find blob with biggest area
        #biggest rectangle in the image must be sudoku square
        self.biggest = None
        self.maxArea = 0
        for i in self.contours:
            area = cv2.contourArea(i)
            if area > 50000: #50000 is an estimated value for the kind of blob we want to evaluate
                peri = cv2.arcLength(i,True)
                approx = cv2.approxPolyDP(i,0.02*peri,True)
                if area > self.maxArea and len(approx)==4:
                    self.biggest = approx
                    self.maxArea = area
                    best_cont = i
        if self.maxArea > 0:
            status.noDetect = 0 #reset
            status.detect += 1   
            self.reorder() #reorder self.biggest
        else:
            status.noDetect += 1
            if status.noDetect == 7:
                print "No sudoku puzzle detected!"
            if status.noDetect > 15:
                status.restart = True
		print "Restart"
        if status.detect == 1:
                status.puzzleFound = True
                print "Sudoku puzzle detected!"
        if status.beginSolver == False or self.maxArea == 0:
            cv2.imshow('sudoku', self.captured)
            key = cv2.waitKey(10)
            if key == 27:
		cap.release()
                sys.exit()


    def reorder(self):
        #reorders the points obtained from finding the biggest rectangle
        #[top-left, top-right, bottom-right, bottom-left]
        a = self.biggest.reshape((4,2))
        b = np.zeros((4,2),dtype = np.float32)
     
        add = a.sum(1)
        b[0] = a[np.argmin(add)] #smallest sum
        b[2] = a[np.argmax(add)] #largest sum
             
        diff = np.diff(a,axis = 1) #y-x
        b[1] = a[np.argmin(diff)] #min diff
        b[3] = a[np.argmax(diff)] #max diff
        self.biggest = b

    def formSudokuCells(self):
        #create 100 points using "biggest" and simple gridding algorithm,
        #these 100 points define the grid of the sudoku puzzle
        #topLeft-topRight-bottomRight-bottomLeft = "biggest"
        b = np.zeros((100,2),dtype = np.float32)
        c_sqrt=10
        if self.biggest == None:
            self.biggest = [[0,0],[640,0],[640,480],[0,480]]
        tl,tr,br,bl = self.biggest[0],self.biggest[1],self.biggest[2],self.biggest[3]
        for k in range (0,100):
            i = k%c_sqrt #column
            j = k/c_sqrt #row
            ml = [tl[0]+(bl[0]-tl[0])/9*j,tl[1]+(bl[1]-tl[1])/9*j] 
            mr = [tr[0]+(br[0]-tr[0])/9*j,tr[1]+(br[1]-tr[1])/9*j]
            self.mat.itemset((k,0),ml[0]+(mr[0]-ml[0])/9*i) #x
            self.mat.itemset((k,1),ml[1]+(mr[1]-ml[1])/9*i) #y
        self.reshape = self.mat.reshape((c_sqrt,c_sqrt,2))

    def perspectiveTrans(self):
        #take distorted image and warp to flat square for clear OCR reading
        mask = np.zeros((self.gray.shape),np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
        close = cv2.morphologyEx(self.gray,cv2.MORPH_CLOSE,kernel)
        division = np.float32(self.gray)/(close)
        result = np.uint8(cv2.normalize(division,division,0,255,cv2.NORM_MINMAX))
        result = cv2.cvtColor(result,cv2.COLOR_GRAY2BGR)
        output = np.zeros((450,450,3),np.uint8)
        c_sqrt=10
        for i,j in enumerate(self.mat):
            ri = i/c_sqrt
            ci = i%c_sqrt
            if ci != c_sqrt-1 and ri != c_sqrt-1:
                source = self.reshape[ri:ri+2, ci:ci+2 , :].reshape((4,2))
                dest = np.array( [ [ci*450/(c_sqrt-1),ri*450/(c_sqrt-1)],[(ci+1)*450/(c_sqrt-1),
                            ri*450/(c_sqrt-1)],[ci*450/(c_sqrt-1),(ri+1)*450/(c_sqrt-1)],
                            [(ci+1)*450/(c_sqrt-1),(ri+1)*450/(c_sqrt-1)] ], np.float32)
                trans = cv2.getPerspectiveTransform(source,dest)
                warp = cv2.warpPerspective(result,trans,(450,450))
                output[ri*450/(c_sqrt-1):(ri+1)*450/(c_sqrt-1) , ci*450/(c_sqrt-1):(ci+1)*450/
                       (c_sqrt-1)] = warp[ri*450/(c_sqrt-1):(ri+1)*450/(c_sqrt-1) ,
                        ci*450/(c_sqrt-1):(ci+1)*450/(c_sqrt-1)].copy()
        output_backup = np.copy(output)
        self.output = output
        self.outputBackup = output_backup

    def virtualImage(self,puzzle):
        tsize = (math.sqrt(self.maxArea))/400
        w = int(20*tsize)
        h = int(25*tsize)
 	if puzzle.check == 1: #Have solution
	    if puzzle.hintCell != 0: #Not solve yet
	        i = puzzle.hintCell[0]*10 + puzzle.hintCell[1]
                x = int(self.mat.item(i,0)+8*tsize)
                y = int(self.mat.item(i,1)+8*tsize)
                if i%10!=9 and i/10!=9:
                    xc = puzzle.hintCell[1]
                    yc = puzzle.hintCell[0]
                    if puzzle.solvedSDKarray[yc,xc]!=0 and puzzle.SDKarray[yc,xc]==0:
                       string = str(puzzle.solvedSDKarray[yc][xc])
                       cv2.putText(self.captured,string,(x+w/4,y+h),0,tsize,(0,255,255),2)

	elif puzzle.check == 0: #Do not have solution
	    wrongGrid = self.biggest.reshape(4,1,2).astype(int)
            cv2.fillConvexPoly(self.captured,wrongGrid,(50,100,220))

	else: # Invalid
	    row1,column1 = puzzle.dictToArr(puzzle.check[0])
	    cell1 = row1*10+column1
            px1 = int(self.mat.item(cell1,0))
            py1 = int(self.mat.item(cell1,1))
            px1Boundary = int(self.mat.item(cell1+11,0))
            py1Boundary = int(self.mat.item(cell1+11,1))

	    row2,column2 = puzzle.dictToArr(puzzle.check[1])
	    cell2 = row2*10+column2
            px2 = int(self.mat.item(cell2,0))
            py2 = int(self.mat.item(cell2,1))
            px2Boundary = int(self.mat.item(cell2+11,0))
            py2Boundary = int(self.mat.item(cell2+11,1))

	    cv2.rectangle(self.captured,(px1,py1),(px1Boundary,py1Boundary),(0,0,95),8)
	    cv2.rectangle(self.captured,(px2,py2),(px2Boundary,py2Boundary),(0,0,95),8)

        cv2.imshow('sudoku',self.captured)
        key = cv2.waitKey(10)
        if key==27:
	    cap.release()
            sys.exit()
	elif key == ord('s'):
	    print "OK"
	    cv2.imwrite("Image/"+str(datetime.datetime.now())+".png",self.captured)
	    print "done"

class OCRmodelClass:
    #this class defines the data used for OCR,
    #and the associated methods for performing OCR
    def __init__(self):
        samples = np.loadtxt('generalsamples.data',np.float32)
        responses = np.loadtxt('generalresponses.data',np.float32)
        responses = responses.reshape((responses.size,1))
        #.model uses kNearest to perform OCR
        self.model = cv2.KNearest()
        self.model.train(samples,responses)
        #.iterations contains information on what type of morphology to use
        self.iterations = [-1,0]
        self.lvl = 0 #index of .iterations
	self.fontsize = 1
        
    def OCR(self,status,image,puzzle):
        #preprocessing for OCR
        #convert image to grayscale
        gray = cv2.cvtColor(image.output, cv2.COLOR_BGR2GRAY)
        #noise removal with gaussian blur
        gray = cv2.GaussianBlur(gray,(5,5),0)
        image.outputGray = gray
        
        #attempt to read the image with 4 different morphology values and find the best result
        self.success = [0,0]
        self.errors = [0,0]
        for self.lvl in self.iterations:
            image.output = np.copy(image.outputBackup)
            self.OCR_read(status,image,puzzle)

        best = 8
        for i in range(2):
            if self.success[i] > best and self.errors[i] == 0:
                best = self.success[i]
                ibest = i
        print "success:",self.success
        print "errors:",self.errors
        
        if best==8:
            print "ERROR - OCR FAILURE"
            status.restart = True
        else:
            print "final morph erode iterations:",self.iterations[ibest]
            image.output = np.copy(image.outputBackup)
            self.lvl = self.iterations[ibest]
            self.OCR_read(status,image,puzzle)

    def OCR_read(self,status,image,puzzle):
        #perform actual OCR using kNearest model
        thresh = cv2.adaptiveThreshold(image.outputGray,255,1,1,7,2)
        if self.lvl >= 0:
            morph = cv2.morphologyEx(thresh,cv2.MORPH_ERODE,None,iterations = self.lvl)
        elif self.lvl == -1:
            morph = cv2.morphologyEx(thresh,cv2.MORPH_DILATE,None,iterations = 1)

        thresh_copy = morph.copy()
        #thresh2 changes after findContours
        contours,hierarchy = cv2.findContours(morph,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        thresh = thresh_copy

        puzzle.SDKarray = np.zeros((9,9),np.uint8)
	self.fontsize+=1
        # testing section
        for cnt in contours:
            if cv2.contourArea(cnt)>20:
                [x,y,w,h] = cv2.boundingRect(cnt)
                if  h>20 and h<40 and w>8 and w<40:
                    if w<20:
                        diff = 20-w
                        x -= diff/2
                        w += diff
                    sudox = x/50
                    sudoy = y/50
                    cv2.rectangle(image.output,(x,y),(x+w,y+h),(0,0,255),2)
                    #prepare region of interest for OCR kNearest model
                    roi = thresh[y:y+h,x:x+w]
		    try:
                    	roismall = cv2.resize(roi,(25,35))
		    except cv2.error:
			 continue
			 print "~.~"
                    roismall = roismall.reshape((1,875))
                    roismall = np.float32(roismall)
                    #find result
                    retval, results, neigh_resp, dists = self.model.find_nearest(roismall, k = 1)
                    #check for read errors
                    if results[0][0]!=0:
                        string = str(int((results[0][0])))
                        if puzzle.SDKarray[sudoy,sudox]==0:
                            puzzle.SDKarray[sudoy,sudox] = int(string)
                        else:
                            self.errors[self.lvl+1]=-2 #double read error
                        self.success[self.lvl+1]+=1
                        cv2.putText(image.output,string,(x,y+h),0,1.4,(255,0,0),2)
                    else:
                        self.errors[self.lvl+1]=-3 #read zero error
                    

class solverStatusClass:
    #this class defines the status of the main loop
    def __init__(self):
        #.beginSolver becomes true when the puzzle is completely captured and ready to solve
        self.beginSolver = False
        #.puzzleFound becomes true when the puzzle is thought to be found but not yet read with OCR
        self.puzzleFound = False
        #.puzzleRead becomes true when OCR has confirmed the puzzle
        self.puzzleRead = False
        #.restart becomes true when the main loop needs to restart
        self.restart = False
        #.completed becomes true when the puzzle has been solved
        self.completed = False
        #.number of times imageClass.captureImage() detects no puzzle
        self.noDetect = 0
        #.number of times imageClass.captureImage() detects a puzzle
        self.detect = 0




reader = OCRmodelClass()
while True:
    status = solverStatusClass()
    while status.beginSolver == False:
	status = solverStatusClass()
        puzzle = puzzleStatusClass()
        image = imageClass()
        print "Waiting for puzzle..."
        while status.puzzleFound == False:
            image.captureImage(status)
            if status.restart == True:
                break
        while status.puzzleRead == False and status.puzzleFound == True:
            image.captureImage(status)
            image.formSudokuCells()
	    image.perspectiveTrans()
            reader.OCR(status,image,puzzle)
            if status.restart == True:
                print "Restarting..."
                break
            elif np.array_equal(puzzle.SDKarray,puzzle.last):
                status.beginSolver = True
                status.puzzleRead = True
            else:
                puzzle.last = np.copy(puzzle.SDKarray)
    puzzle.getSDKdict()
    puzzle.checkSDK()
    puzzle.getHintCell()
    status.completed = True
    while status.completed == True:
        image.captureImage(status)
        if status.restart == True:
            print "Restarting..."
            break
        if image.maxArea > 0:
            image.formSudokuCells()
            image.virtualImage(puzzle)
	
