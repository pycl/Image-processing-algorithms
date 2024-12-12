import cv2
import numpy as np
import re

def bits2string(bits, reverse_bits=False):
    #Преобразование списка битов в ASCII-строку
    chars = []
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i+8]
        if len(byte_bits) < 8:
            break
        byte_value = 0
        if reverse_bits:
            # Рассматривать bit[0] как самый старший бит
            for b in byte_bits:
                byte_value = (byte_value << 1) | b
        else:
            # Рассматривать bit[0] как самый младший бит
            for idx, b in enumerate(byte_bits):
                byte_value |= (b << idx)
        chars.append(chr(byte_value))
        result = ''.join(chars)
    return result

def extract_lsb_channel(img, channel_index=0):
    """
    Извлечение битовой последовательности из младших значащих разрядов (LSB) указанного канала.
    channel_index: 0-B, 1-G, 2-R 
    """
    height, width, _ = img.shape
    bits = []
    for row in range(height):
        for col in range(width):
            pixel = img[row, col]
            bit = pixel[channel_index] & 1 #Получение младшего бита
            bits.append(bit)
    return bits

def find_password(s):
    """С помощью регулярных выражений выполняется поиск паролей, 
    состоящих из 8 символов, включающих буквы и цифры, в строке."""
    match = re.search(r'[A-Za-z0-9]{8}', s)
    if match:
        return match.group(0)
    return None


if __name__ == '__main__':
    img_path = 'test.png' 
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("Не получится загрузить изображение, проверьте путь к файлу.")

    channels = ['B', 'G', 'R']
    found_passwords = []

    # Попробуем извлечь младший значащий бит (LSB) из каждого канала
    for ch_index, ch_name in enumerate(channels):
        bits = extract_lsb_channel(img, ch_index)

        for reverse_flag in [False, True]:
            secret_str = bits2string(bits, reverse_bits=reverse_flag)

            # Поиск пароля в строке
            pwd = find_password(secret_str)
            if pwd:
                found_passwords.append((ch_name, reverse_flag, pwd))


    if found_passwords:
        for ch, rev, p in found_passwords:
            print(f"Channel: {ch}, Reverse:{rev}, Password: {p}")
    else:
        print("Ничего не найдено")