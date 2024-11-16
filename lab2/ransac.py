import random
import match as m
from svd import matrix_multiply,matrix_transpose,svd_solve


def estimate_fundamental_matrix(points1, points2):
    """оценка фундаментальной матрицы"""
    if len(points1) < 8 or len(points2) < 8:
        return None
    
    # построение матрицыA
    A = []
    for (x1, y1), (x2, y2) in zip(points1, points2):
        A.append([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])
    
    try:
        # использование SVD для решения Af = 0
        U, S, V = svd_solve(A)
        
        # взятие последней строки V (соответствующей минимальному сингулярному значению) в качестве элементов F
        F = [V[-1][i:i+3] for i in range(0, 9, 3)]
        
        # использование SVD для решения и принудительное задание ранга 2 (основная фундаментальная матрица должна быть ранга 2)
        U_f, S_f, V_f = svd_solve(F)
        S_f[2][2] = 0  # установка минимального сингулярного значения в 0
        
        # восстановление F
        F = matrix_multiply(matrix_multiply(U_f, S_f), matrix_transpose(V_f))
        
        # нормализация F
        norm = sum(sum(x*x for x in row) for row in F) ** 0.5
        F = [[x/norm for x in row] for row in F]
        
        return F
    except:
        return None
    
#вычисление расстояния от точки до линии
def apply_fundamental_matrix(F, point1, point2):
    #используется для проверки того, соответствуют ли пара точек соответствующей фундаментальной матрице   
    x1, y1 = point1 # точка в изображении 1
    x2, y2 = point2 # точка в изображении 2
    vec1 = [x1, y1, 1]
    vec2 = [x2, y2, 1]

    # вычисление F * vec1 для получения линии в изображении 2
    # эта линия находится в изображении 2
    # теоретически point2 должна находиться на этой линии
    line = [sum(F[i][j] * vec1[j] for j in range(3)) for i in range(3)]
    
    # вычисление расстояния от point2 до этой линии
    numerator = abs(sum(line[i] * vec2[i] for i in range(3)))
    denominator = (line[0]**2 + line[1]**2)**0.5
    #возвращает расстояние от point2 до линии
    #когда возвращается float('inf'), эта пара точек будет обработана как выброс(outlier),так как ее расстояние больше любого конечного порога
    return numerator/denominator if denominator > 1e-10 else float('inf')

def ransac_fundamental_matrix(matches, keypoints1, keypoints2, threshold=3.0, max_iterations=1000):
    """Оценка фундаментальной матрицы с использованием RANSAC для фильтрации выбросов"""
    if len(matches) < 8:
        return matches
    
    best_inliers = []
    best_F = None
    
    #извлечение координат точек из matches
    points1 = [(keypoints1[m['kp1_idx']]['x'], keypoints1[m['kp1_idx']]['y'])  # точка в изображении 1
               for m in matches]
    points2 = [(keypoints2[m['kp2_idx']]['x'], keypoints2[m['kp2_idx']]['y'])  # соответствующая точка в изображении 2
               for m in matches]
    
    for _ in range(max_iterations):
        # случайный выбор 8 пар точек
        if len(matches) < 8:
            continue
        
        sample_indices = random.sample(range(len(matches)), 8)
        sample_points1 = [points1[i] for i in sample_indices]
        sample_points2 = [points2[i] for i in sample_indices]
        
        # оценка фундаментальной матрицы
        F = estimate_fundamental_matrix(sample_points1, sample_points2)
        if F is None:
            continue
            
        # вычисление внутренних точек
        # в идеальном случае
        # 1. point2 должна находиться строго на линии
        # 2. расстояние от точки до линии должно быть равно 0
        # error = 0  является идеальным случаем, но из-за различных причин возникают ошибки, 
        # поэтому нужно установить порог,чтобы пара точек считалась внутренней точкой(inlier),если расстояние меньше порога
        current_inliers = []
        for idx, (pt1, pt2) in enumerate(zip(points1, points2)):
            error = apply_fundamental_matrix(F, pt1, pt2)
            if error < threshold:
                current_inliers.append(matches[idx])
        
        # обновление лучшего результата
        if len(current_inliers) > len(best_inliers):
            best_inliers = current_inliers
            best_F = F
    
    return best_inliers

def match_descriptors_with_ransac(descriptors1, descriptors2, keypoints1, keypoints2):
    # сначала выполняется обычное сопоставление
    initial_matches = m.match_descriptors(descriptors1, descriptors2)
    print(f"начальное количество совпадений:{len(initial_matches)}")

    # использование RANSAC для фильтрации совпадений
    filtered_matches = ransac_fundamental_matrix(
        initial_matches,
        keypoints1,
        keypoints2,
        threshold=10,
        max_iterations=1000
    )
    print(f"после RANSAC: {len(filtered_matches)}")

    return filtered_matches