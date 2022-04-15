import cv2
import matplotlib.pyplot as plt

img = cv2.imread("C:/Image Processing/Python Projects IP/Template Matching/cat.jpg",0)

template = cv2.imread("C:/Image Processing/Python Projects IP/Template Matching/cat_face.jpg",0) 

height,width = template.shape;


methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    
    method = eval(meth) #'cv2.TM_CCOEFF' -> cv2.TM_CCOEFF
    
    res = cv2.matchTemplate(img, template, method)
    
    min_value, max_value, min_location, max_lcoation = cv2.minMaxLoc(res)
    
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        
        top_left = min_location
        
    else:
        
        top_left = max_lcoation
        
    bottom_right = (top_left[0] + width, top_left[1] + height)
    
    cv2.rectangle(img, top_left,bottom_right, 255, 2)
    
    plt.figure()
    plt.subplot(121), plt.imshow(res, cmap = "gray")
    plt.title("Eşleşen Sonuç"), plt.axis("off")
    plt.subplot(122), plt.imshow(img, cmap = "gray")
    plt.title("Tespit edilen Sonuç"), plt.axis("off")
    plt.suptitle(meth)
    
    plt.show()
    

