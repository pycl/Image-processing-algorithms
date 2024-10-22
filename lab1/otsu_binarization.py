#Использование метода Оцу для бинаризации цветного изображения и возврат бинаризованного изображения.
#image: трехмерный список, представляющий цветное изображение. Каждый элемент `image[i][j]` — это список `[R, G, B]`.
#Возвращается двумерный список, представляющий бинаризованное изображение, где 1 обозначает передний план, а 0 — фон.
def otsu_binarization(image):
    # Получение размеров изображения
    height = len(image)
    width = len(image[0])
    #преобразование цветного изображения в оттенки серого
    grayscale = RGB2GRAY(image)

    histogram = [0] * 256  # Инициализация гистограммы из 256 нулей
    for i in range(height):
            # Извлечение значений RGB
        for j in range(width):
            gray_value = grayscale[i][j]
            # Увеличение счетчика соответствующего значения серого
            histogram[gray_value] += 1  

    #реализация метода Оцу для поиска оптимального порога
    total_pixels = height * width
    sum_total = 0  # Общая сумма значений серого всех пикселей
    for t in range(256):
        sum_total += t * histogram[t]

    sumB = 0  # Кумулятивная сумма значений серого для фона
    wB = 0    # Вес фона (количество пикселей)
    maximum_between_class_variance = 0
    optimal_threshold = 0

    for t in range(256):
        wB += histogram[t]  
        if wB == 0:
            continue  # Если нет пикселей, принадлежащих этому классу, пропустить
        wF = total_pixels - wB   #Вес класса переднего плана
        if wF == 0:
            break # Если все пиксели принадлежат фону, остановить цикл
        sumB += t * histogram[t]  # Кумулятивная сумма значений серого для фона
        mB = sumB / wB             # Среднее значение серого для класса фона
        mF = (sum_total - sumB) / wF   # Среднее значение серого для класса переднего плана
        between_class_variance = wB * wF * (mB - mF) ** 2
        # Обновление максимальной межклассовой дисперсии и оптимального порога
        if between_class_variance > maximum_between_class_variance:
            maximum_between_class_variance = between_class_variance
            optimal_threshold = t
#создание бинаризованного изображения с использованием оптимального порога
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