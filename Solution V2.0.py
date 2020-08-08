#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

#get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:



##test = cv2.imread("./matchingPics/mf00.JPG")
files = glob.glob("./matchingPics/*.jpg")

#load all the image filenames
#the widefield image is stored in wf and the zoom images in mf
mf = []
wf = None
for f in files:
    if os.path.basename(f)[:2] == "mf":
        mf.append(f)
    else:
        wf = f


# In[4]:


#function that masks an image given a tuple
#most masks have 1600 and 1000 for width and height, except the last two
def rect2img(img,x,y,w=1600,h=1000):
    m = np.zeros(base_g.shape)
    m[y:y+h,x:x+w] = img[y:y+h,x:x+w]
    return m

rects = []
rects.append((95,1945))
rects.append((510,1990))
rects.append((1015,1855))
rects.append((1500, 1700))
rects.append((1860, 1885))
rects.append((2355, 1850))
rects.append((2950, 1850))
rects.append((3425, 1820))
rects.append((3920, 1830))
rects.append((4160, 1775))
rects.append((4650, 1765,800,500))
rects.append((5150, 1775,800,500))


# In[21]:


def BFMMatch(td,bd,k):
    #Build a brute force matcher. NORM_HAMMING is recommended for orb descriptors. crossCheck set to False so KNN matching works later
    BFM = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    #instead of just bruteforce matching, knn matching works better.
    #k=2 so we can apply a threshold filter
    #matches = BFM.match(td,bd)
    matches = BFM.knnMatch(td,bd,k=k)
    print("KNN matched: ",len(matches))

    #filter the matches
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    print("Good matches: ",len(matches))
    return good


# In[31]:


GMS_COUNT = 150
def GMS_LOOP(query_shape,base_shape,tk,bk,matches):
    t=7
    GMS = cv2.xfeatures2d.matchGMS(query_shape,base_shape,tk,bk,matches,withScale=True,withRotation=False,thresholdFactor=t)
    #reduce threshold value for GMS till we have enough matches
    while len(GMS) < GMS_COUNT:
        print("GMS: ",len(GMS))
        t = t - 1
        if t < 0:
            break
        GMS = cv2.xfeatures2d.matchGMS(query_shape,base_shape,tk,bk,matches,withScale=True,withRotation=False,thresholdFactor=t)
    
    #accounts for if GMS breaks and isn't returning matches even at a 0 threshold.
    if t < 0 and len(GMS) == 0:
        print("GMS is zero. Skipping")
        return None
    print("GMS: ",len(GMS))
    return GMS


# In[37]:


AVERAGE_AREA = 430*300
def transform(GMS,shape, query, base, res):
    #Probably more important, but unused.
    #can refine homography by taking top 10 or 20 of GMS after sorting
    GMS = sorted(GMS,key=lambda x:x.distance)

    #get "from" and "to" coordinates
    src_pts = np.float32([ tk[m.queryIdx].pt for m in GMS])
    dst_pts = np.float32([ bk[m.trainIdx].pt for m in GMS])
    
    
    loopGMS = True
    t = 6.0
    
    #loop and make the ransac threshold stricter until we get a rectangle
    
    while loopGMS:
        #find the transformation matrix based on these points
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,t)

        #make a simple box to find the four points of our destination
        h,w = shape
        pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)

        #transform the box so it fits over the destination
        dst = cv2.perspectiveTransform(pts,M)

        #pull top left corner and bottom right corner
        p1 = (int(dst[0][0][0]),int(dst[0][0][1]))
        p2 = (int(dst[2][0][0]),int(dst[2][0][1]))

        #find the width and the height
        w = abs(p2[0] - p1[0])
        h = abs(p2[1] - p1[1])
        print(w,h)
        print(w*h)
        print(AVERAGE_AREA)
        #rudementary check if the result is a rectangle
        if w > h and w*h>AVERAGE_AREA:
            #condition met for our rectangles
            #break the loop
            loopGMS = False
        else:
            t=t-1
        
        if t == -1:
            return None,None

    try:
        #resize the original query image so it fits in the destination
        query_resized = cv2.resize(query,(w,h),cv2.INTER_CUBIC)
        
        #overlay the query image on the base image
        base[p1[1]:p1[1]+h,p1[0]:p1[0]+w,:] = query_resized
        
        #overlay the query image on the resultant image
        res[p1[1]:p1[1]+h,p1[0]:p1[0]+w,:] = query_resized
        
        return base, res
    except:
        print("Failure at: ", str(ft), str(sf))
        return None,None

    


# In[38]:



#loop through all our zoom images

sfs = [1.5,1.6,1.7,1.8,1.9,2.0]
fts = [10000,20000,30000,40000,50000]
sf = sfs[0]
ft = fts[4]
res = cv2.imread(wf)
for i in range(len(mf)):
    
    
    #load query image and reset base image
    #query image is the zoom image
    #base image is the wide field image
    query = cv2.imread(mf[i])
    base = cv2.imread(wf)
    #base= np.copy(res)

    #process the query image first
    #convert to grayscale
    #equalize the histogram
    #scale it down. Each call to pyrDown effective halves the image, so three calls gives us an eighth of our original image
    #size becomes 750,500 which is close enough to the actual size on the map

    query_g = cv2.cvtColor(query,cv2.COLOR_RGB2GRAY)
    query_g = cv2.equalizeHist(query_g)
    query_g = cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(query_g)))


    #process the base image
    #convert to grayscale
    #equalize the histogram
    #mask it according to the rectangles

    base_g = cv2.cvtColor(base,cv2.COLOR_RGB2GRAY)
    base_g = cv2.equalizeHist(base_g)
    clipped_base = rect2img(base_g,*rects[i]).astype(np.uint8)


    #build an orb feature extractor
    #we want as many features as possible so "fastThreshold" is set to 0 and "edgeThreshold" is set low
    orb = cv2.ORB_create(nfeatures=ft,scaleFactor=sf,nlevels=5,edgeThreshold=15,firstLevel=0,WTA_K=2,fastThreshold=0)


    #generate keypoints and descriptors for the base and query images
    bk,bd = orb.detectAndCompute(clipped_base,None)
    tk,td = orb.detectAndCompute(query_g,None)
    print("ORB: ","bk: ", len(bk),"tk",len(tk))

    
    #match and filter our descriptors
    matches = BFMMatch(td,bd,2)

    #filter through GMS
    GMS = GMS_LOOP(query_g.shape[:2],base_g.shape[:2],tk,bk,matches)
    
    if GMS is None:
        continue
    
    #transform and overlay
    base,res = transform(GMS,query_g.shape[:2],query,base,res)
    
    if base is None or res is None:
        continue
    
    cv2.imwrite("block" + str(i) + ".jpg",base)
cv2.imwrite("res - " + str(ft) + "- " + str(sf) + ".jpg",res)

