import cv2
import numpy as np
import matplotlib.pyplot as plt

"""plot a list of images"""
def plot_list(lst):
    num_rows = len(lst)

    for i, word in enumerate(lst):
        ax = plt.subplot(num_rows, 1, i+1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(word)
        
    plt.show()
    
    
""" returns list of text lines from a grayscale image"""    
def get_line_boxes(grayed_image,
                  AREA_THRESHOLD=600,
                  GAUSSIAN_BLUR_KERNEL=(5,5),
                  DILATION_KERNEL_SIZE=(7,1),
                  THRESH_NEIGHBORHOOD_SIZE=5,
                  THRESH_TUNE=3,
                  MIN_HEIGHT=10,
                  MIN_WIDTH=10):

    # Load image, grayscale, Gaussian blur, adaptive threshold
    blur = cv2.GaussianBlur(grayed_image, GAUSSIAN_BLUR_KERNEL, 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,THRESH_NEIGHBORHOOD_SIZE, THRESH_TUNE)

    # save an image to draw the lines and to extract lines from
    image_with_boxes = thresh.copy()
    image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
    image_with_boxes = cv2.bitwise_not(image_with_boxes)
    clean_image = image_with_boxes.copy()

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (DILATION_KERNEL_SIZE)) # makes rectangular kernel of the given size
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        

    y_list = []
    lines = []
    ROI_number = 0
    good_contours = []
    for c in cnts:
        area = cv2.contourArea(c)
        
        # only draws boxes around blobs with a certain area
        if area < AREA_THRESHOLD:
            continue
            
        x,y,w,h = cv2.boundingRect(c)
        
        # minimum ratio - want horizontal boxes
        if h > 2 * w:
            continue
            
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            continue
            
        good_contours.append(c)
            
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (36,255,12), 1) 
        ROI = clean_image[y:y+h, x:x+w]
        y_list.append(y)
        lines.append(ROI)

    ordered_list = []
        
    y_list = np.asarray(y_list)
    for i in range(len(lines)):
        minYIndex = np.argmin(y_list)
        ordered_list.append(lines[minYIndex])
        y_list[minYIndex] = 10000
        
    assert len(lines) == len(contours)
    return ordered_list, thresh, dilate, image_with_boxes, good_contours


"""reprocess image for line granularity
    takes the grayscale image and contours returned by get_line_boxes()"""
def get_lines(grayed_image, contours,
             GAUSSIAN_BLUR_KERNEL=(3,3),
             THRESH_NEIGHBORHOOD_SIZE=3,
             THRESH_TUNE=2):
    
    y_list = []
    lines = []
    for c in contours:
        area = cv2.contourArea(c)
        x,y,w,h = cv2.boundingRect(c)
        
        ROI = grayed_image[y:y+h, x:x+w]
        
        # individually blur and threshold each line
        blur = cv2.GaussianBlur(ROI, GAUSSIAN_BLUR_KERNEL, 0)
        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV,THRESH_NEIGHBORHOOD_SIZE, THRESH_TUNE)
        clean_line = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB)
        clean_line = cv2.bitwise_not(clean_line)
        
        y_list.append(y)
        lines.append(clean_line)

    ordered_list = []
        
    y_list = np.asarray(y_list)
    for i in range(len(lines)):
        minYIndex = np.argmin(y_list)
        ordered_list.append(lines[minYIndex])
        y_list[minYIndex] = 10000
        
    return ordered_list #, blur, thresh, clean_image


"""segments and returns list of words from the given line of text"""
def get_words(line, WORD_AREA_THRESHOLD=1, WORD_DILATION_KERNEL_SIZE=(3, 3)):

    image_with_boxes = line.copy()
    gray = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,11,50)

    # Dilate to combine adjacent text contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, WORD_DILATION_KERNEL_SIZE) 
    dilate = cv2.dilate(thresh, kernel, iterations=1) # can tune the number of iterations

    # Find contours, highlight text areas, and extract ROIs
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    word_list = []
    x_list = []
    for c in cnts:
        area = cv2.contourArea(c)
        # maybe add an area threshold back in
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (36,255,12), 1) 
        ROI = line[y:y+h, x:x+w]
        x_list.append(x)
        word_list.append(ROI)

    ordered_word_list = []
        
    x_list = np.asarray(x_list)
    for i in range(len(word_list)):
        minXIndex = np.argmin(x_list)
        ordered_word_list.append(word_list[minXIndex])
        x_list[minXIndex] = 10000
        
    return ordered_word_list, thresh, image_with_boxes, dilate