import cv2
import numpy as np
import os
import time

import threading


#利用opencv偵測手的邊緣
minValue = 70

x0 = 400
y0 = 200
height = 200
width = 200

saveImg = False
guessGesture = False
visualize = False

lastgesture = -1

kernel = np.ones((15,15),np.uint8)
kernel2 = np.ones((1,1),np.uint8)

#定義一個5x5的橢圓結構元素另有十字形结构（MORPH_CROSS)和矩形(MORPH_RECT）
skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

# Which mask mode to use BinaryMask, SkinMask (True|False) OR BkgrndSubMask ('x' key)
binaryMode = True
bkgrndSubMode = False
mask = 0
bkgrnd = 0
counter = 0
# This parameter controls number of image samples to be taken PER gesture
numOfSamples = 301
gestname = ""
path = ""
mod = 0

banner =  '''\nWhat would you like to do ?
    1- Use pretrained model for gesture recognition & layer visualization
    2- Train the model (you will require image samples for training under .\imgfolder)
    3- Visualize feature maps of different layers of trained model
    4- Exit	
    '''


#%%
def saveROIImg(img):
    global counter, gestname, path, saveImg
    if counter > (numOfSamples - 1):
        # Reset the parameters
        saveImg = False
        gestname = ''
        counter = 0
        return
    
    counter = counter + 1
    name = gestname + str(counter)
    print("Saving img:",name)
    cv2.imwrite(path+name + ".jpg", img)
    time.sleep(0.04 )


#%%



#%%
def skinMask(frame, x0, y0, width, height, framecount, plot):
    global guessGesture, visualize, mod, lastgesture, saveImg
    # HSV values
    low_range = np.array([0, 50, 80])
    upper_range = np.array([30, 200, 255])
    #(image,頂點,對角座標,顏色,thickness, 線種類,shift)
    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    #roi = cv2.UMat(frame[y0:y0+height, x0:x0+width])
    roi = frame[y0:y0+height, x0:x0+width]
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    #Apply skin color range
    mask = cv2.inRange(hsv, low_range, upper_range)
    
    mask = cv2.erode(mask, skinkernel, iterations = 2)
    mask = cv2.dilate(mask, skinkernel, iterations = 1)
    
    #blur
    mask = cv2.GaussianBlur(mask, (15,15), 1)
    
    #bitwise and mask original frame
    res = cv2.bitwise_and(roi, roi, mask = mask)
    # color to grayscale
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    
    if saveImg == True:
        saveROIImg(res)
    elif guessGesture == True and (framecount % 5) == 4:
        #res = cv2.UMat.get(res)
        t = threading.Thread(target=myNN.guessGesture, args = [mod, res])
        t.start()
    elif visualize == True:
        layer = int(input("Enter which layer to visualize "))
        cv2.waitKey(0)
        myNN.visualizeLayers(mod, res, layer)
        visualize = False
    
    return res


	
	
	
#%%
def Main():
    global guessGesture, visualize, mod, binaryMode, bkgrndSubMode, mask, takebkgrndSubMask, x0, y0, width, height, saveImg, gestname, path
    quietMode = False
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 0.5
    fx = 10
    fy = 350
    fh = 18

        
        
    ## Grab camera input
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)

    # set rt size as 640x480
    ret = cap.set(3,640)
    ret = cap.set(4,480)

    framecount = 0
    fps = ""
    start = time.time()

    plot = np.zeros((512,512,3), np.uint8)
    
	
	#optical flow
    ret, old_frame = cap.read()
    old_frame_optical= old_frame[y0:y0+height, x0:x0+width]
    roi_old = skinMask(old_frame, x0, y0, width, height, framecount, plot)
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.9,
                           minDistance = 50,
                           blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    
    p0 = cv2.goodFeaturesToTrack(roi_old, mask = None, **feature_params)

    # Create a mask image for drawing purposes   
    mask_optical = np.zeros_like(old_frame_optical)
	
    while(True):
        ret, frame = cap.read()
        max_area = 0
        
        frame = cv2.flip(frame, 3)
        frame = cv2.resize(frame, (640,480))
                      
        if ret == True:
            frame_optical= frame[y0:y0+height, x0:x0+width]
            #if binaryMode == True:
            roi = skinMask(frame, x0, y0, width, height, framecount, plot)


            
            framecount = framecount + 1
            end  = time.time()
            timediff = (end - start)
            if( timediff >= 1):
                #timediff = end - start
                fps = 'FPS:%s' %(framecount)
                start = time.time()
                framecount = 0
        cv2.putText(frame,fps,(10,20), font, 0.7,(0,255,0),2,1)
        cv2.putText(frame,'Options:',(fx,fy), font, 0.7,(0,255,0),2,1)
        #cv2.putText(frame,'b - Toggle Binary/SkinMask',(fx,fy + fh), font, size,(0,255,0),1,1)
        #cv2.putText(frame,'x - Toggle Background Sub Mask',(fx,fy + 2*fh), font, size,(0,255,0),1,1)		
        #cv2.putText(frame,'g - Toggle Prediction Mode',(fx,fy + 3*fh), font, size,(0,255,0),1,1)
        cv2.putText(frame,'q - Toggle Quiet Mode',(fx,fy + 4*fh), font, size,(0,255,0),1,1)
        cv2.putText(frame,'n - To enter name of new gesture folder',(fx,fy + 5*fh), font, size,(0,255,0),1,1)
        cv2.putText(frame,'s - To start capturing new gestures for training',(fx,fy + 6*fh), font, size,(0,255,0),1,1)
        cv2.putText(frame,'ESC - Exit',(fx,fy + 7*fh), font, size,(0,255,0),1,1)
        cv2.putText(frame,'w - delet track',(fx,fy + 3*fh), font, size,(0,255,0),1,1)
        
		
        
        #p0 = cv2.goodFeaturesToTrack(roi_old, mask = None, **feature_params)
		# calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(roi_old, roi, p0, None, **lk_params)
        if p1 is None:
            p0 = cv2.goodFeaturesToTrack(roi_old, mask = None, **feature_params)
            p1, st, err = cv2.calcOpticalFlowPyrLK(roi_old, roi, p0, None, **lk_params)
                
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        
        
        
        
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
          a,b = new.ravel()
          c,d = old.ravel()
          mask_optical = cv2.line(mask_optical, (a,b),(c,d), color[i].tolist(), 2)
          frame_optical = cv2.circle(frame_optical,(a,b),5,color[i].tolist(),-1)
        img_optical = cv2.add(frame_optical,mask_optical)
        
		
        ## If enabled will stop updating the main openCV windows
        ## Way to reduce some processing power :)
        if not quietMode:
            cv2.imshow('Original',frame)
            cv2.imshow('ROI', roi)
            cv2.namedWindow('optical', cv2.WINDOW_NORMAL)
            cv2.resizeWindow("optical", 640, 480)
            cv2.imshow('optical',img_optical)
            #cv2.imshow('mask', mask)
            #plot = np.zeros((512,512,3), np.uint8)
        
		
        key = cv2.waitKey(5) & 0xff
        
        ## Use Esc key to close the program
        if key == 27:
            break
        
        ## This option is not yet complete. So disabled for now
        ## Use v key to visualize layers
        #elif key == ord('v'):
        #    visualize = True

        ## Use i,j,k,l to adjust ROI window
        elif key == ord('i'):
            y0 = y0 - 5
        elif key == ord('k'):
            y0 = y0 + 5
        elif key == ord('j'):
            x0 = x0 - 5
        elif key == ord('l'):
            x0 = x0 + 5

        ## Quiet mode to hide gesture window
        elif key == ord('q'):
            quietMode = not quietMode
            print("Quiet Mode - {}".format(quietMode))

        ## Use s key to start/pause/resume taking snapshots
        ## numOfSamples controls number of snapshots to be taken PER gesture
        elif key == ord('s'):
            saveImg = not saveImg
            
            if gestname != '':
                saveImg = True
            else:
                print("Enter a gesture group name first, by pressing 'n'")
                saveImg = False
        
        ## Use n key to enter gesture name
        elif key == ord('n'):
            gestname = input("Enter the gesture folder name: ")
            try:
                os.makedirs(gestname)
            except OSError as e:
                # if directory already present
                if e.errno != 17:
                    print('Some issue while creating the directory named -' + gestname)
            
            path = "./"+gestname+"/"
        elif key == ord('w'):
            mask_optical = np.zeros_like(old_frame_optical)
            mask_optical = cv2.line(mask_optical, (a,b),(c,d), color[i].tolist(), 2)
	
		# Now update the previous frame and previous points
        roi_old = roi.copy()
        p0 = good_new.reshape(-1,1,2)

    #Realse & destroy
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    Main()