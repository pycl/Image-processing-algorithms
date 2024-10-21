def otsu_binarization(image):
    height = len(image)
    width = len(image[0])

    grayscale = RGB2GRAY(image)

    
    histogram = [0] * 256  
    for i in range(height):
        for j in range(width):
            gray_value = grayscale[i][j]
            histogram[gray_value] += 1  

    
    total_pixels = height * width
    sum_total = 0  
    for t in range(256):
        sum_total += t * histogram[t]

    sumB = 0  
    wB = 0    
    maximum_between_class_variance = 0
    optimal_threshold = 0

    for t in range(256):
        wB += histogram[t]  
        if wB == 0:
            continue  
        wF = total_pixels - wB  
        if wF == 0:
            break 
        sumB += t * histogram[t]  
        mB = sumB / wB  
        mF = (sum_total - sumB) / wF  
        between_class_variance = wB * wF * (mB - mF) ** 2
        if between_class_variance > maximum_between_class_variance:
            maximum_between_class_variance = between_class_variance
            optimal_threshold = t

    binary_image = []
    for i in range(height):
        binary_row = []
        for j in range(width):
            if grayscale[i][j] > optimal_threshold:
                binary_row.append(1)  
            else:
                binary_row.append(0) 
        binary_image.append(binary_row)

    return binary_image

def RGB2GRAY(image):
    height = len(image)
    width = len(image[0])
    grayscale = []
    for i in range(height):
        grayscale_row = []
        for j in range(width):
            R,G,B = image[i][j]
            gray = int(0.299*R + 0.589*G + 0.114*B)
            grayscale_row.append(gray)
        grayscale.append(grayscale_row)
    return grayscale