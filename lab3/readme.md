## Основная логика Алгоритма:
1. Использование OpenCV для чтения изображения.
2. Извлечение информации из младших значащих битов всех пикселей в трех каналах изображения.
3. Попытка преобразовать извлеченные последовательности битов в ASCII-строку. При этом используются два способа:
    - Первый извлеченный бит рассматривается как младший бит байта (reverse_bits=False).
    - Первый извлеченный бит рассматривается как старший бит байта (reverse_bits=True).
4.Применение регулярного выражения  к полученной строке для поиска 8-символьных фрагментов, состоящих из букв и цифр.
5.Если найдено совпадение, вывод возможного пароля.

## После тестирования вывод следующий:
Channel: B, Reverse:True, Password: sRy9ysBO
Channel: R, Reverse:True, Password: 07C2Vfxh