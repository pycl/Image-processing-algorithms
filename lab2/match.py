import keypoint as kp
import ransac as rs

def process_images(image_filenames):
    all_images = []
    all_keypoints = []
    all_descriptors = []

    for file_name in image_filenames:
        #read_image включает в себя функцию преобразования изображения в оттенки серого
        image_gray = kp.read_image(file_name)
        all_images.append(image_gray)

        Ix,Iy = kp.compute_gradient(image_gray)
        R = kp.harris_corner_detection(image_gray)
        threshold = max([max(row) for row in R if row]) *0.01
        keypoints = kp.non_maximum_suppression(R,threshold)
        magnitude,orientation = kp.compute_gradient_magnitude_orientation(Ix,Iy)
        kp.compute_keypoint_orientations(keypoints,magnitude,orientation)
        descriptors = kp.compute_sift_descriptors(image_gray,keypoints,magnitude,orientation)

        all_keypoints.append(keypoints)
        all_descriptors.append(descriptors)

    return all_images,all_keypoints,all_descriptors  

#Вычисление евклидова расстояния между двумя дескрипторами
def euclidean_distance(desc1, desc2):
    distance = 0
    for a, b in zip(desc1, desc2):
        distance += (a - b) ** 2
    return distance ** 0.5

def match_descriptors(descriptors1, descriptors2):
    matches = []
    for idx1, kp1 in enumerate(descriptors1):
        descriptor1 = kp1['descriptor']
        min_distance = float('inf')        # ближайшее расстояние
        second_min_distance = float('inf') # второе ближайшее расстояние
        best_match_idx = -1

        for idx2, kp2 in enumerate(descriptors2):
            descriptor2 = kp2['descriptor']
            distance = euclidean_distance(descriptor1, descriptor2)
            if distance < min_distance:
                # ближайшее расстояние становится вторым ближайшим
                second_min_distance = min_distance
                # обновление ближайшего расстояния
                min_distance = distance
                best_match_idx = idx2
            elif distance < second_min_distance:
                second_min_distance = distance

        # применение теста отношения
        if min_distance < 0.85 * second_min_distance:
            matches.append({
                'kp1_idx': idx1,
                'kp2_idx': best_match_idx,
                'distance': min_distance
            })

    return matches


def match_features_between_images(all_descriptors, all_keypoints):
    matches_dict = {}
    num_images = len(all_descriptors)
    
    for i in range(num_images):
        for j in range(i + 1, num_images):
            descriptors1 = all_descriptors[i]
            descriptors2 = all_descriptors[j]
            keypoints1 = all_keypoints[i]
            keypoints2 = all_keypoints[j]
            
            # применение RANSAC для сопоставления дескрипторов
            matches = rs.match_descriptors_with_ransac(
                descriptors1, 
                descriptors2,
                keypoints1,
                keypoints2
            )
            # сохранение сопоставлений между изображениями i и j
            matches_dict[(i, j)] = matches
    
    return matches_dict

def build_tracks(all_keypoints, matches_dict):
    # создание словаря для каждого изображения
    keypoint_ids = [{} for _ in all_keypoints]
     # счетчик идентификаторов траекторий
    track_id = 0
    
    # перебор всех пар изображений и сопоставлений
    for (i, j), matches in matches_dict.items():
        for match in matches:
            # индекс ключевой точки в i изображении
            idx_i = match['kp1_idx']
            # индекс ключевой точки в j изображении
            idx_j = match['kp2_idx']
            
            # получение текущего идентификатора траектории ключевой точки
            id_i = keypoint_ids[i].get(idx_i)
            id_j = keypoint_ids[j].get(idx_j)
            
            if id_i is None and id_j is None:
                # нет идентификаторов траекторий, назначение нового идентификатора траектории
                keypoint_ids[i][idx_i] = track_id
                keypoint_ids[j][idx_j] = track_id
                track_id += 1
            elif id_i is not None and id_j is None:
                # ключевая точка i уже имеет идентификатор траектории, назначение идентификатора траектории ключевой точке j
                keypoint_ids[j][idx_j] = id_i
            elif id_i is None and id_j is not None:
                # ключевая точка j уже имеет идентификатор траектории, назначение идентификатора траектории ключевой точке i
                keypoint_ids[i][idx_i] = id_j
            elif id_i != id_j:
                # требуется объединение идентификаторов траекторий
                old_id = id_j # идентификатор траектории в j изображении
                new_id = id_i # идентификатор траектории в i изображении
                for k in range(len(all_keypoints)):
                    for kp_idx in keypoint_ids[k]:
                        if keypoint_ids[k][kp_idx] == old_id:
                            keypoint_ids[k][kp_idx] = new_id
    return keypoint_ids


def draw_matches_multiple_images(all_images, all_keypoints, matches_dict, max_matches=100):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # вычисление общей ширины и максимальной высоты изображения
    total_width = 0
    max_height = 0
    image_offsets = []  # сохранение смещения каждого изображения

    for img in all_images:
        height = len(img)
        width = len(img[0])
        image_offsets.append(total_width)  # сохранение смещения каждого изображения
        total_width += width                # суммирование общей ширины
        if height > max_height:             # поиск максимальной высоты
            max_height = height

    # создание изображения для склейки
    stitched_image = [[0] * total_width for _ in range(max_height)]

    # склеивание всех изображений
    for idx, img in enumerate(all_images):
        offset = image_offsets[idx]
        height = len(img)
        width = len(img[0])
        for y in range(height):
            for x in range(width):
                stitched_image[y][x + offset] = img[y][x]

    # рисование изображения
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.imshow(stitched_image, cmap='gray')

    # определение списка цветов
    colors = ['yellow', 'cyan', 'magenta', 'green', 'blue', 'red', 'orange', 'purple']

    # рисование сопоставлений
    for idx, ((i, j), matches) in enumerate(matches_dict.items()):
        # сортировка сопоставлений по расстоянию, выбор лучших max_matches сопоставлений
        matches_sorted = sorted(matches, key=lambda x: x['distance'])
        matches_to_draw = matches_sorted[:max_matches]

        offset_i = image_offsets[i]
        offset_j = image_offsets[j]
        keypoints_i = all_keypoints[i]
        keypoints_j = all_keypoints[j]

        color = colors[idx % len(colors)]  # выбор цвета

        for match in matches_to_draw:
            idx_i = match['kp1_idx']
            idx_j = match['kp2_idx']
            x1 = keypoints_i[idx_i]['x'] + offset_i
            y1 = keypoints_i[idx_i]['y']
            x2 = keypoints_j[idx_j]['x'] + offset_j
            y2 = keypoints_j[idx_j]['y']

            # рисование ключевых точек
            ax.scatter([x1, x2], [y1, y2], c='r', s=5)

            # рисование линии
            line = Line2D([x1, x2], [y1, y2], linewidth=0.5, color=color)
            ax.add_line(line)

    plt.axis('off')
    plt.show()
