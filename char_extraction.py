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
    
    
""" returns list of text lines from a grayscale image with fixed default params"""    
def __get_fixed_line_contours(grayed_image,
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
        
    assert len(lines) == len(good_contours)
    return ordered_list, thresh, dilate, image_with_boxes, good_contours


'''gets the line contours using a dynamically determined dilation kernel size
 returns (lines, thresholded image, dilated image, image_w_boxes, contours) if
 a good kernel found, otherwise returns all None -> examine the output
 before proceeding
 don't use the lines from this function, pass the contours to get_lines()'''
def old_get_adaptive_line_contours(img):
    line_dilation_kernel_sizes = [3, 5, 7, 9, 11, 13, 15]

    all_num_lines = []
    best_stddev = None
    best_kernel = None
    best_num_lines = None
    for l in line_dilation_kernel_sizes:
        _, _, _, _, contours = __get_fixed_line_contours(img, DILATION_KERNEL_SIZE=(l,1))
        lines = get_lines(img, contours)
        
        num_lines = len(lines)
        all_num_lines.append(num_lines)
        
        heights = []
        for line in lines:
            s = line.shape
            heights.append(s[0])

        stddev = np.std(heights)
        if best_stddev is None:
            best_stddev = stddev
            best_kernel = l
            best_num_lines = num_lines
        elif stddev <= best_stddev: # take the smallest stddev in line height, preferring larger kernels
            best_stddev = stddev
            best_kernel = l
            best_num_lines = num_lines
                    
    # find the best one
    med = np.median(all_num_lines)
    mod = max(all_num_lines, key = all_num_lines.count)
    
    if 0.9*med <= best_num_lines <= 1.1*med or 0.9*mod <= best_num_lines <= 1.1*mod:
        lines, thr, dil, img_w_boxes, contours = __get_fixed_line_contours(img, DILATION_KERNEL_SIZE=(best_kernel,1))
        return lines, thr, dil, img_w_boxes, contours
    else:
        return None, None, None, None, None # did not find good segementation
    

# try different params
def get_adaptive_line_contours(img):
    line_dilation_kernel_sizes = [3, 5, 7, 9, 11, 13, 15]
    size_to_num_lines = dict()
    size_to_height_stddev = dict()
    all_num_lines = []
    
    for k in line_dilation_kernel_sizes:
        lines, thresh, dilate, _, contours = __get_fixed_line_contours(img, DILATION_KERNEL_SIZE=(k,1))
        lines = get_lines(img, contours)
        num_lines = len(lines)
        size_to_num_lines[k] = num_lines
        all_num_lines.append(num_lines)
        
        heights = []
        for line in lines:
            s = line.shape
            heights.append(s[0])
            
        stddev = np.std(heights)
        size_to_height_stddev[k] = stddev
        
    # find the best one
    med = np.median(all_num_lines)
    mod = max(all_num_lines, key = all_num_lines.count)
    
    # sort kernels by increasing stddev of line heights
    ks = [l for l, val in sorted(size_to_height_stddev.items(), key=lambda item: item[1])]
    
    for k in ks:
        num_lines = size_to_num_lines[k]
        if 0.9*med <= num_lines <= 1.1*med or 0.9*mod <= num_lines <= 1.1*mod:
            lines, thr, dil, img_w_boxes, contours = __get_fixed_line_contours(img, DILATION_KERNEL_SIZE=(k,1))
            return lines, thr, dil, img_w_boxes, contours
    
    return None, None, None, None, None # no segmentation found


"""reprocess image for word granularity
    takes the grayscale image and contours returned by get_adaptive_line_contours()
    returns list of 3D line images to be used in get_adaptive_words()"""
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


'''adaptively segments a list of lines from a page into words using
dynamically determined word dilation kernel
returns a list of all words on the page (2D and binary images)'''
def get_adaptive_words(lines_of_page):
    word_dilation_kernel_sizes = [3, 4, 5, 6]

    best_stddev = None
    best_kernel = None
    best_num_lines = None
    for w1 in word_dilation_kernel_sizes:
            for w2 in word_dilation_kernel_sizes: 
                all_words = []
                heights = []
                widths = []
                for line in lines_of_page:
                    word_list, thresh, _, dilate = __get_words(line, WORD_DILATION_KERNEL_SIZE=(w1, w2))
                    all_words.extend(word_list)
                    heights.extend([w.shape[0] for w in word_list])
                    widths.extend([w.shape[1] for w in word_list])
                    
                stddev_h = np.std(heights)
                
                if best_stddev is None:
                    best_stddev = stddev_h
                    best_kernel = (w1, w2)
                elif stddev_h <= best_stddev: # take the smallest stddev in line height
                    best_stddev = stddev_h
                    best_kernel = (w1, w2)

    all_words = []
    for line in lines_of_page:
        word_list, _, _, _ = __get_words(line, WORD_DILATION_KERNEL_SIZE=best_kernel)
        all_words.extend(word_list)
    return all_words


"""segments and returns list of words from the given line of text"""
def __get_words(line, WORD_AREA_THRESHOLD=20,
              WORD_DILATION_KERNEL_SIZE=(4, 3),
             MIN_HEIGHT=5,
                  MIN_WIDTH=5):

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
        if area < WORD_AREA_THRESHOLD:
            continue
            
        x,y,w,h = cv2.boundingRect(c)
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            continue
            
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (36,255,12), 1) 
        ROI = gray[y:y+h, x:x+w]
        x_list.append(x)
        word_list.append(ROI)

    ordered_word_list = []
        
    x_list = np.asarray(x_list)
    for i in range(len(word_list)):
        minXIndex = np.argmin(x_list)
        ordered_word_list.append(word_list[minXIndex])
        x_list[minXIndex] = 10000
        
    return ordered_word_list, thresh, image_with_boxes, dilate