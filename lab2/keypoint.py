import matplotlib.pyplot as plt
import math
#Преобразование RGB-изображения в  оттенки серого
def RGB2GRAY(image):
    gray=[]
    height = len(image)
    width = len(image[0])
    for i in range(height):
        row=[]
        for j in range(width):
            r,g,b = image[i][j]
            I = 0.2989*r +0.587*g + 0.114*b
            row.append(I)
        gray.append(row)
    return gray

def read_image(filename):
    image =plt.imread(filename)
    image = image.tolist()
    if len(image[0][0]) == 3:
        image_gray = RGB2GRAY(image)
    else:
        image_gray = image
    return image_gray


def compute_gradient(image_gray):
    height = len(image_gray)
    width = len(image_gray[0])

    # Оператор Собеля
    Gx = [[-1,0,1],
          [-2,0,2],
          [-1,0,1]]
    
    Gy = [[-1,-2,-1],
          [0,0,0],
          [1,2,1]]
    
    # Инициализация выходного массива, сохраняющего такой же размер, как и входное изображение
    Ix = [[0 for _ in range(width)] for _ in range(height)]
    Iy = [[0 for _ in range(width)] for _ in range(height)]

    # padding = (kernel_size - 1) // 2 = 1
    
    # Добавление padding
    padded_image = []
    # Верхний padding
    padded_image.append([0] * (width + 2))
    # Среднее содержимое
    for row in image_gray:
        padded_image.append([0] + row + [0])
    # Нижний padding
    padded_image.append([0] * (width + 2))
  
    
    # Вычисление градиента
    for i in range(height):
        for j in range(width):
            sum_x = 0
            sum_y = 0
            for x in range(3):
                for y in range(3):
                    pixel_val = padded_image[i+x][j+y]
                    sum_x += pixel_val * Gx[x][y]
                    sum_y += pixel_val * Gy[x][y]
            Ix[i][j] = sum_x
            Iy[i][j] = sum_y

    return Ix, Iy

def compute_gradient_products(Ix,Iy):
    height = len(Ix)
    width = len(Ix[0])

    Ixx = []
    Iyy = []
    Ixy = []

    for i in range(height):
        row_xx = []
        row_yy = []
        row_xy = []
        for j in range(width):
            ix = Ix[i][j]
            iy = Iy[i][j]
            ixx = ix * ix
            iyy = iy * iy
            ixy = ix* iy
            row_xx.append(ixx)
            row_yy.append(iyy)
            row_xy.append(ixy)
        Ixx.append(row_xx)
        Ixy.append(row_xy)
        Iyy.append(row_yy)
    return Ixx,Iyy,Ixy

def gaussian_kernal(size,sigma):
    kernel = []
    center = size // 2
    sum_val = 0

    for i in range(size):
        row = []
        for j in range(size):
            x = j-center
            y = i-center
            # x и y — координаты относительно центра, поэтому необходимо вычесть center.
            exponent = -(x**2 + y**2) / (2*sigma**2)
            value = (1/(2*math.pi*sigma**2))*math.exp(exponent)
            row.append(value)
            sum_val += value
        kernel.append(row)
    # Нормализация ядра
    for i in range(size):
        for j in range(size):
            kernel[i][j] /=sum_val
    
    return kernel

def convolve(image, kernel):
    height = len(image)
    width = len(image[0])
    kernel_size = len(kernel)
    padding = kernel_size // 2  # Расчет необходимого размера padding
    
    # Создание заполненного изображения
    padded_image = []
    # Верхний padding
    for _ in range(padding):
        padded_image.append([0] * (width + 2*padding))
    # Среднее содержимое
    for row in image:
        padded_image.append([0] * padding + row + [0] * padding)
    # Нижний padding
    for _ in range(padding):
        padded_image.append([0] * (width + 2*padding))
    
    # Инициализация результирующего массива (такого же размера, как исходное изображение)
    result = [[0 for _ in range(width)] for _ in range(height)]
    
    # Выполнение свертки
    for i in range(height):
        for j in range(width):
            sum_val = 0
            for m in range(kernel_size):
                for n in range(kernel_size):
                    pixel = padded_image[i+m][j+n] * kernel[m][n]
                    sum_val += pixel
            result[i][j] = sum_val
    
    return result

def compute_R(Sxx,Syy,Sxy,k=0.04):
    height = len(Sxx)
    width = len(Sxx[0])

    R = [[0 for _ in range(width)]for _ in range(height)]

    for i in range(height):
        for j in range(width):
            det = Sxx[i][j] * Syy[i][j] - Sxy[i][j]**2
            trace = Sxx[i][j] + Syy[i][j]
            R[i][j] = det - k*(trace**2)
    return R

#Объединение нескольких функций, реализующих обнаружение углов Харриса
def harris_corner_detection(image_gray,k=0.04,gussian_size=3,sigma=1.5):
    Ix,Iy = compute_gradient(image_gray)
    Ixx,Iyy,Ixy = compute_gradient_products(Ix,Iy)
    kernel = gaussian_kernal(gussian_size,sigma)
    Sxx = convolve(Ixx,kernel)
    Syy = convolve(Iyy,kernel)
    Sxy = convolve(Ixy,kernel)
    R = compute_R(Sxx,Syy,Sxy,k)
    return R

def non_maximum_suppression(R,threshold):
    height = len(R)
    width = len(R[0])
    corners = []

    for i in range(1,height-1):
        for j in range(1,width-1):
            #Использование немаксимального подавления для проверки этого пикселя 3x3 
            #R[i][j] — это координаты центра пикселя 3x3 матрицы
            #Если текущий пиксель в пределах области имеет максимальное значение отклика, считается, что он является углом.
            if R[i][j] > threshold:
                local_max = True
                for m in range(-1,2):
                    for n in range(-1,2):
                        if R[i+m][j+n] > R[i][j]:
                            local_max = False
                            break
                    if not local_max:
                        break
                #Добавление координат (i, j) и соответствующего значения отклика R[i][j] в словарь corners
                if local_max:
                    corners.append({'x':j,'y':i,'response':R[i][j]})
    return corners
#Вычисление градиента изображения и его величины и ориентации
def compute_gradient_magnitude_orientation(Ix,Iy):
    height = len(Ix)
    width = len(Ix[0])
    magnitude = [[0 for _ in range(width)]for _ in range(height)]
    orientation = [[0 for _ in range(width)]for _ in range(height)]

    for y in range(height):
        for x in range(width):
            gx = Ix[y][x]
            gy = Iy[y][x]
            magnitude[y][x] = (gx**2+gy**2)**0.5
            orientation[y][x] = (math.degrees(math.atan2(gy,gx))+360)%360
    return magnitude,orientation
#Назначение главной ориентации для каждого ключевого точки:
def compute_keypoint_orientations(keypoints,magnitude,orientation):
    for kp in keypoints:
        x = kp['x']
        y = kp['y']
        #Определение радиуса области
        radius = 8
        hist_bins = 36
        hist = [0]*hist_bins

        for i in range(-radius,radius+1):
            for j in range(-radius,radius+1):
                xj = x+j
                yi = y+i
                #Ограничение диапазона
                if 0 <= xj <len(magnitude[0]) and 0 <= yi <len(magnitude):
                    #Вычисление расстояния с гауссовым весом
                    distance = (i ** 2 + j ** 2) ** 0.5
                    if distance > radius:
                        continue
                    weight = math.exp(-(distance ** 2)) / (2*(radius / 2)**2)
                    #Получение величины градиента и направления
                    mag = magnitude[yi][xj] * weight
                    angle = orientation[yi][xj]
                    #Вычисление индекса гистограммы
                    bin_idx = int(angle / 10) % hist_bins
                    hist[bin_idx] += mag
        #Поиск максимального значения в гистограмме
        max_bin_value = max(hist)
        max_bin_index = hist.index(max_bin_value)
        kp_angle = max_bin_index*10 #Каждый бит 10°
        kp['angle'] = kp_angle

#Вычисление SIFT-дескриптора
def compute_sift_descriptors(image_gray, keypoints, magnitude, orientation):
    height = len(image_gray)
    width = len(image_gray[0])
    radius = 8
    descriptors = []

    # Проверка размеров массивов
    assert len(magnitude) == height and len(magnitude[0]) == width, "Magnitude array size mismatch"
    assert len(orientation) == height and len(orientation[0]) == width, "Orientation array size mismatch"

    for kp in keypoints:
        x = kp['x']
        y = kp['y']
        
        # Пропуск ключевых точек, близких к краю
        if (x < radius or x >= width - radius or 
            y < radius or y >= height - radius):
            continue
            
        descriptor = [0] * 128
        kp_angle = kp.get('angle', 0)

        for sub_y in range(-radius, radius + 1):
            for sub_x in range(-radius, radius + 1):
                xi = x + sub_x
                yj = y + sub_y
                if 0 <= xi < width and 0 <= yj < height:
                    bin_x = int((sub_x + radius) / 4)
                    bin_y = int((sub_y + radius) / 4)
                    if 0 <= bin_x < 4 and 0 <= bin_y < 4:
                        mag = magnitude[yj][xi]
                        #Вычисление относительного угла направления пикселя
                        angle = orientation[yj][xi] - kp_angle
                        angle = (angle + 360) % 360

                        bin_orientation = int(angle / 45) % 8
                        index = (bin_y * 4 + bin_x) * 8 + bin_orientation
                        descriptor[index] += mag

        # Нормализация
        norm = sum([v ** 2 for v in descriptor]) ** 0.5
        if norm > 0:
            descriptor = [v / norm for v in descriptor]
        
        kp['descriptor'] = descriptor
        descriptors.append(kp)

    return descriptors




