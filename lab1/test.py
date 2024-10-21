import unittest
import random
from otsu_binarization import otsu_binarization

class TestOtsuBinarization(unittest.TestCase):
    def test_high_contrast_image(self):
        """
        Тестирование алгоритма на изображении с четким контрастом между передним планом и фоном.
        """
        # Создаем изображение с высоким контрастом
        height, width = 10, 10
        image = []
        for i in range(height):
            row = []
            for j in range(width):
                if i < height // 2:
                    pixel = [0, 0, 0]  #  Черный пиксель
                else:
                    pixel = [255, 255, 255]  # Белый пиксель
                row.append(pixel)
            image.append(row)

        # Применим метод Оцу
        binary_image = otsu_binarization(image)

        # Проверить, что алгоритм правильно разделяет две области
        for i in range(height):
            for j in range(width):
                if i < height // 2:
                    self.assertEqual(binary_image[i][j], 0)
                else:
                    self.assertEqual(binary_image[i][j], 1)

    def test_low_contrast_image(self):
        """
        Тестирование алгоритма на изображениях с низкой контрастностью.
        """
        height, width = 10, 10
        # создание изображения с пиксельными значениями в узком диапазоне серого
        image = []
        for i in range(height):
            row = []
            for j in range(width):
                value = 120 + random.randint(-5, 5)  #  Значения серого от 115 до 125
                pixel = [value, value, value]
                row.append(pixel)
            image.append(row)

        binary_image = otsu_binarization(image)
   
        first_value = binary_image[0][0]
        for row in binary_image:
            for value in row:
                self.assertEqual(value, first_value)


    def test_noisy_image(self):
        """
        Тестирование алгоритма на изображении со случайным шумом.
        """
        random.seed(0)  # Для воспроизводимости
        height, width = 10, 10
        image = []
        for _ in range(height):
            row = []
            for _ in range(width):
                # Сгенерируем случайные значения RGB
                pixel = [random.randint(0, 255) for _ in range(3)]
                row.append(pixel)
            image.append(row)

  
        binary_image = otsu_binarization(image)

        # Поскольку изображение зашумлено, мы не можем предсказать точные значения,
        #  но мы можем проверить, что все выходные значения равны 0 или 1
        for i in range(height):
            for j in range(width):
                self.assertIn(binary_image[i][j], [0, 1])

    def test_blank_image(self):
        """
        Тестирование алгоритма на пустом изображении, где все пиксели имеют одинаковый цвет.
        """
        height, width = 10, 10
        image = [[[128, 128, 128] for _ in range(width)] for _ in range(height)]  # Gray image

        binary_image = otsu_binarization(image)

        # Выход должен быть все нули или все единицы
        first_value = binary_image[0][0]
        for row in binary_image:
            for value in row:
                self.assertEqual(value, first_value)

    def test_gradient_image(self):
        """
        Тестирование работы алгоритма на изображениях с градиентом от чёрного к белому.
        """
        height, width = 10, 10
        image = []
        for i in range(height):
            row = []
            for j in range(width):
                # Генерация градиентных значений от 0 до 255
                value = int((i * width + j) / (height * width - 1) * 255)
                pixel = [value, value, value]
                row.append(pixel)
            image.append(row)

        binary_image = otsu_binarization(image)

        #  Чтобы проверить, содержит ли результат бинаризации 0 и 1
        unique_values = set()
        for row in binary_image:
            unique_values.update(row)
        self.assertTrue(0 in unique_values and 1 in unique_values)

    def test_multiple_thresholds_image(self):
        """
        Тестирование работы алгоритма на изображениях с несколькими областями разной степени серого.
        """
        height, width = 10, 10
        image = []
        for i in range(height):
            row = []
            for j in range(width):
                if j < width // 3:
                    pixel = [50, 50, 50]    # Тёмно-серый
                elif j < 2 * width // 3:
                    pixel = [128, 128, 128]  # Средне-серый 
                else:
                    pixel = [200, 200, 200]  # Светло-серый
                row.append(pixel)
            image.append(row)

        binary_image = otsu_binarization(image)

       # Проверка, правильно ли результат бинаризации разделил области
        for i in range(height):
            for j in range(width):
                if j < width // 3:
                    expected_value = 0
                else:
                    expected_value = 1
                self.assertEqual(binary_image[i][j], expected_value)

    
    def test_salt_and_pepper_noise(self):
        """
        Тестирование работы алгоритма на изображениях с шумом «соль и перец»
        """
        random.seed(2)
        height, width = 10, 10
        image = [[[128, 128, 128] for _ in range(width)] for _ in range(height)]  # Базовое серое изображение

        # Добавление шума «соль и перец»
        num_noisy_pixels = int(0.1 * height * width)  #10% шумовых пикселей
        for _ in range(num_noisy_pixels):
            i = random.randint(0, height - 1)
            j = random.randint(0, width - 1)
            if random.choice([True, False]):
                image[i][j] = [0, 0, 0]     # Чёрный шум (перец)
            else:
                image[i][j] = [255, 255, 255]  # Белый шум (соль)

        
        binary_image = otsu_binarization(image)

       # Проверка, согласована ли классификация большинства пикселей изображения
        counts = {0: 0, 1: 0}
        for row in binary_image:
            for value in row:
                counts[value] += 1

        # Поскольку большинство пикселей серые, алгоритм должен отнести их к одному классу
        self.assertTrue(counts[0] > counts[1] or counts[1] > counts[0])


    def test_checkerboard_pattern(self):
        """
        Тестирование работы алгоритма на изображениях с шахматной доской.
        """
        height, width = 8, 8  
        image = []
        for i in range(height):
            row = []
            for j in range(width):
                if (i + j) % 2 == 0:
                    pixel = [0, 0, 0]     
                else:
                    pixel = [255, 255, 255]  
                row.append(pixel)
            image.append(row)

        
        binary_image = otsu_binarization(image)

        # Проверка, соответствует ли результат бинаризации узору шахматной доски
        for i in range(height):
            for j in range(width):
                expected_value = 0 if (i + j) % 2 == 0 else 1
                self.assertEqual(binary_image[i][j], expected_value)

    def test_image_with_text(self):
        """
        Тестирование работы алгоритма на изображениях, имитирующих текст.
        """
        height, width = 10, 10
        image = []
        for i in range(height):
            row = []
            for j in range(width):
                if (27<= i <= 32 and 15 <=j <=62) | (35 <=j <= 39 and 33<= i <= 90):
                    pixel = [0, 0, 0]  # Чёрные области текста
                else:
                    pixel = [255, 255, 255]  # Белый фон
                row.append(pixel)
            image.append(row)
        binary_image = otsu_binarization(image)
        # Проверка, правильно ли текстовые области распознаны как передний план
        for i in range(height):
            for j in range(width):
                if (27<= i <= 32 and 15 <=j <=62) | (35 <=j <= 39 and 33<= i <= 90):
                    self.assertEqual(binary_image[i][j], 0)  # Передний план (текст)
                else:
                    self.assertEqual(binary_image[i][j], 1)  # фон

    def test_large_image(self):
        """
        Тестирование производительности и корректности алгоритма на больших изображениях.
        """
        height, width = 100, 100
        image = []
        for i in range(height):
            row = []
            for j in range(width):
                #  Создание узора шахматной доски
                if (i // 10 + j // 10) % 2 == 0:
                    pixel = [50, 50, 50]  #  Тёмные квадраты
                else:
                    pixel = [200, 200, 200]  # Светлые квадраты
                row.append(pixel)
            image.append(row)
      
        binary_image = otsu_binarization(image)
     
        for i in range(height):
            for j in range(width):
                expected_value = 0 if (i // 10 + j // 10) % 2 == 0 else 1
                self.assertEqual(binary_image[i][j], expected_value)


if __name__ == '__main__':
    unittest.main()