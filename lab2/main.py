import match as m
import camera_pose as cp
from Visualization import visualize_point_cloud

image_filenames = [f'images/test{i}.jpg' for i in range(1,6)]

# обработка много изображений
all_images, all_keypoints, all_descriptors = m.process_images(image_filenames)

# сопоставление признаков между изображениями
matches_dict = m.match_features_between_images(all_descriptors,all_keypoints)

# построение траекторий признаков
keypoint_ids = m.build_tracks(all_keypoints, matches_dict)


# оценка положения камеры и триангуляция
K = [[6000, 0, 150],
    [0, 6000, 200],
    [0, 0, 1]]  
camera_poses = {0: {'R': [[1,0,0],[0,1,0],[0,0,1]], 't': [[0],[0],[0]]}}  # первая камера как ссылка
point_cloud = []
for (i, j), matches in matches_dict.items():
    # извлечение соответствующих точек
    points1 = [(all_keypoints[i][m['kp1_idx']]['x'], all_keypoints[i][m['kp1_idx']]['y']) for m in matches]
    points2 = [(all_keypoints[j][m['kp2_idx']]['x'], all_keypoints[j][m['kp2_idx']]['y']) for m in matches]
    
    # оценка существенной матрицы
    E = cp.estimate_essential_matrix(points1, points2, K)
    if E is None:
        continue
    
    # извлечение позы камеры из существенной матрицы
    poses = cp.extract_pose_from_essential_matrix(E)
    if poses is None:
        continue
    
    # выбрать правильную позу камеры
    R, t = cp.select_correct_pose(poses, points1, points2, K)
    if R is None or t is None:
        continue
    camera_poses[j] = {'R': R, 't': t}
    
    # триангуляция трехмерных точек
    for idx in range(len(points1)):
        p1 = [[points1[idx][0]], [points1[idx][1]], [1]]
        p2 = [[points2[idx][0]], [points2[idx][1]], [1]]
        P = cp.triangulate_point(p1, p2, R, t)
        if P is not None:
            point_cloud.append(P)

print(f"Всего {len(all_images)} изображений")
total_tracks = set()
for kp_dict in keypoint_ids:
    total_tracks.update(kp_dict.values())
print(f"Всего {len(total_tracks)} траекторий признаков")

# визуализация сопоставлений
m.draw_matches_multiple_images(all_images, all_keypoints, matches_dict)
visualize_point_cloud(point_cloud)