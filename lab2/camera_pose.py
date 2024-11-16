from svd import svd_solve,matrix_multiply,matrix_transpose
from ransac import estimate_fundamental_matrix


def estimate_essential_matrix(points1, points2, K):
    """оценка существенной матрицы"""
    #проверка на чистую вращение
    def check_pure_rotation(pts1, pts2, threshold=5.0):
        mean_motion = sum(((x1-x2)**2 + (y1-y2)**2)**0.5 
                         for (x1,y1), (x2,y2) in zip(pts1, pts2)) / len(pts1)
         # если средний перемещение меньше порога, считается чистой вращение
        return mean_motion < threshold
    
    is_pure_rotation = check_pure_rotation(points1, points2)
    
    if is_pure_rotation:
        # вернуть нулевую матрицу как существенную матрицу
        # потому что:
        # 1. при чистой вращении t=0
        # 2. существенная матрица E = [t]× R
        # 3. когда t=0, E=0
        return [[0,0,0], [0,0,0], [0,0,0]] 
    # сначала оценить фундаментальную матрицу
    F = estimate_fundamental_matrix(points1, points2)
    if F is None:
        return None
    # вычислить существенную матрицу E = K'.T * F * K
    K_transpose = matrix_transpose(K)
    temp = matrix_multiply(K_transpose, F) # K^T * F
    if temp is None:
        return None
    E = matrix_multiply(temp, K) # (K^T * F) * K
    if E is None:
        return None
    
    U, S, V = svd_solve(E)
    if None in (U, S, V):
        return None
    
    # установить сингулярные значения (1,1,0)
    S_new = [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 0]]
    # Пересборка существенной матрицы E = U * S_new * V^T
    temp = matrix_multiply(U, S_new)
    if temp is None:
        return None
    E = matrix_multiply(temp, matrix_transpose(V))
    if E is None:
        return None
    
    return E


def matrix_determinant(R):
    """вычисление определителя матрицы"""
    return (R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) -
            R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0]) +
            R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]))
    

def extract_pose_from_essential_matrix(E):
  
    """извлечение камерной позиции (R,t) из существенной матрицы"""
    
    """возвращает четыре возможных решения [(R1,t1), (R1,-t1), (R2,t2), (R2,-t2)]"""

    # проверка на чистую вращение (E близка к нулевой матрице)
    E_norm = sum(sum(x*x for x in row) for row in E)**0.5
    if E_norm < 1e-3:
        # возвращает единичную матрицу и нулевой вектор переноса
        R = [[1,0,0], [0,1,0], [0,0,1]] # матрицы поворота
        t = [[0], [0], [0]]           # вектора переноса 
        return [(R, t)]
    
    if E is None:
        return None
        
    # SVD разложение
    U, S, Vt = svd_solve(E)
    if None in (U, S, Vt):
        return None
    
    # конструирование матрицы W (W представляет собой 90-градусную вращение)
    # W матрица является вспомогательной матрицей для извлечения вращающей матрицы R из существенной матрицы E.
    # существенная матрица E = [t]× R
    #[t]× -кососимметрическая матрица вектора переноса
    W = [[0, -1, 0],
         [1, 0, 0],
         [0, 0, 1]]
    
    # обеспечение ортогональности U и Vt
    def ensure_orthogonal(R):
        U_r, _, V_r = svd_solve(R)
        if None in (U_r, V_r):
            return None
        return matrix_multiply(U_r, matrix_transpose(V_r))
    
    # конструирование возможных R и t
    temp1 = matrix_multiply(U, W)
    if temp1 is None:
        return None
    R1 = matrix_multiply(temp1, Vt)
    R1 = ensure_orthogonal(R1)
    
    W_t = matrix_transpose(W)
    temp2 = matrix_multiply(U, W_t)
    if temp2 is None:
        return None
    R2 = matrix_multiply(temp2, Vt)
    R2 = ensure_orthogonal(R2)
    
    if None in (R1, R2):
        return None
    
    # t равна последнему столбцу U
    t = [[U[i][2]] for i in range(len(U))]
    
    # обеспечение определителя равным 1
    def fix_rotation(R):
        if R is None:
            return None
        det = matrix_determinant(R)
        if det < 0:
            return [[x * -1 for x in row] for row in R]
        return R
    
    R1 = fix_rotation(R1)
    R2 = fix_rotation(R2)
    
    if None in (R1, R2, t):
        return None
    # возвращает все возможные (R,t) комбинации:
    # из этих четырех комбинаций только одна правильная
    # требуется для определения через триангуляцию и проверку глубины
    
    return [(R1, t), (R1, [[-x[0]] for x in t]), 
            (R2, t), (R2, [[-x[0]] for x in t])]

def matrix_inverse(A):
    """вычисление обратной матрицы 3x3"""
    # вычисление определителя
    det = (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
           A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
           A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]))
    if det == 0:
        raise ValueError("матрица необратима")
    inv_det = 1 / det
    # вычисление присоединенной матрицы и умножение на 1/det
    inv = [
        [
            inv_det * (A[1][1] * A[2][2] - A[1][2] * A[2][1]),
            inv_det * (A[0][2] * A[2][1] - A[0][1] * A[2][2]),
            inv_det * (A[0][1] * A[1][2] - A[0][2] * A[1][1])
        ],
        [
            inv_det * (A[1][2] * A[2][0] - A[1][0] * A[2][2]),
            inv_det * (A[0][0] * A[2][2] - A[0][2] * A[2][0]),
            inv_det * (A[0][2] * A[1][0] - A[0][0] * A[1][2])
        ],
        [
            inv_det * (A[1][0] * A[2][1] - A[1][1] * A[2][0]),
            inv_det * (A[0][1] * A[2][0] - A[0][0] * A[2][1]),
            inv_det * (A[0][0] * A[1][1] - A[0][1] * A[1][0])
        ]
    ]
    return inv

def triangulate_point(p1, p2, R, t):
    """
    триангуляция 3D точки из пары соответствующих точек
    
    параметры:
    p1, p2: нормализованные координаты изображения, формат [[x], [y], [1]]
    R: матрица вращения 3x3
    t: вектор переноса [[tx], [ty], [tz]]
    
    возвращает:
    3D точка координаты [[X], [Y], [Z]]
    """
    #конструирование проекционной матрицы первой камеры
    #Первая камера принимается за начало мировой системы координат(отсутствие поворота и переноса)
    P1 = [[1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 1, 0]]
    
    # конструирование проекционной матрицы второй камеры
    P2 = [[R[0][0], R[0][1], R[0][2], t[0][0]],
          [R[1][0], R[1][1], R[1][2], t[1][0]],
          [R[2][0], R[2][1], R[2][2], t[2][0]]]
    
    # конструирование системы уравнений Ax = 0
    A = []
    # 
    # p1 = P1 * X (для первой камеры)
    # p2 = P2 * X (для второй камеры)
    # p1 × (P1 * X) = 0
    # p2 × (P2 * X) = 0
    # уравнение первой камеры
    A.append([
        p1[0][0]*P1[2][0] - P1[0][0],
        p1[0][0]*P1[2][1] - P1[0][1],
        p1[0][0]*P1[2][2] - P1[0][2],
        p1[0][0]*P1[2][3] - P1[0][3]
    ])
    A.append([
        p1[1][0]*P1[2][0] - P1[1][0],
        p1[1][0]*P1[2][1] - P1[1][1],
        p1[1][0]*P1[2][2] - P1[1][2],
        p1[1][0]*P1[2][3] - P1[1][3]
    ])
    
    # уравнение второй камеры
    A.append([
        p2[0][0]*P2[2][0] - P2[0][0],
        p2[0][0]*P2[2][1] - P2[0][1],
        p2[0][0]*P2[2][2] - P2[0][2],
        p2[0][0]*P2[2][3] - P2[0][3]
    ])
    A.append([
        p2[1][0]*P2[2][0] - P2[1][0],
        p2[1][0]*P2[2][1] - P2[1][1],
        p2[1][0]*P2[2][2] - P2[1][2],
        p2[1][0]*P2[2][3] - P2[1][3]
    ])
    
    U, S, V = svd_solve(A)
    if None in (U, S, V):
        return None
    
     # Берем последнюю строку V как решение в однородных координатах
    X = V[-1]
    if abs(X[3]) < 1e-10:
        return None
    
    # Переход от однородных координат к евклидовым путем деления на W (однородная компонента)
    return [[X[i]/X[3]] for i in range(3)]


def select_correct_pose(poses, points1, points2, K):
    """
    выбор правильной позиции из четырех возможных
    
    параметры:
    poses: четыре возможные позиции [(R,t)]
    points1, points2: соответствующие точки
    K: внутренняя матрица камеры
    
    возвращает:
    правильная позиция (R,t)
    """
    if poses is None or points1 is None or points2 is None or K is None:
        return None, None
    
    best_pose = None
    max_positive_depths = 0
    min_reproj_error = float('inf')
    
    for R, t in poses:
        if R is None or t is None:
            continue
            
        positive_depths = 0
        total_error = 0
        valid_points = 0
        
        for (x1, y1), (x2, y2) in zip(points1, points2):
            # преобразование пиксельных координат в нормализованные координаты
            # используя обратную матрицу внутренних параметров K^(-1)
            p1 = matrix_multiply(matrix_inverse(K), [[x1], [y1], [1]])
            if p1 is None:
                continue
            p2 = matrix_multiply(matrix_inverse(K), [[x2], [y2], [1]])
            if p2 is None:
                continue
            
            # Триангуляция 3D точки из пары соответствующих точек
            P = triangulate_point(p1, p2, R, t)
            if P is None:
                continue
            
            # Проверка хиральности (cheirality check):
            # 3D точка должна находиться перед обеими камерами
            if P[2][0] > 0:  # Проверка Z-координаты для первой камеры
                # Преобразование точки в систему координат второй камеры
                # P2 = R*P + t
                P2 = matrix_multiply(R, P)  # Поворот
                if P2 is None:
                    continue
                P2 = [[P2[i][0] + t[i][0]] for i in range(3)]  # Перенос
                
                if P2[2][0] > 0:  # Проверка Z-координаты для второй камеры
                    positive_depths += 1
                    
                    # Вычисление ошибки репроекции:
                    # 1. Проекция 3D точки обратно на изображения
                    p1_proj = [P[0][0]/P[2][0], P[1][0]/P[2][0]]   # (X/Z, Y/Z) для первой камеры
                    p2_proj = [P2[0][0]/P2[2][0], P2[1][0]/P2[2][0]]  # (X/Z, Y/Z) для второй камеры
                    
                    # 2. Вычисление квадрата евклидова расстояния между
                    # спроецированными и исходными точками
                    error1 = ((p1_proj[0] - p1[0][0])**2 + 
                             (p1_proj[1] - p1[1][0])**2)  # для первой камеры
                    error2 = ((p2_proj[0] - p2[0][0])**2 + 
                             (p2_proj[1] - p2[1][0])**2)  # для второй камеры
                    total_error += error1 + error2
                    valid_points += 1
        
        # Если найдены валидные точки для текущей позы
        if valid_points > 0:
            # Вычисляем среднюю ошибку репроекции
            avg_error = total_error / valid_points
            
            # Выбираем лучшую позу по двум критериям:
            if (positive_depths > max_positive_depths or  # Критерий 1: больше точек перед камерами
                (positive_depths == max_positive_depths and  # Критерий 2: при равном количестве точек
                 avg_error < min_reproj_error)):           # выбираем позу с меньшей ошибкой
                max_positive_depths = positive_depths
                min_reproj_error = avg_error
                best_pose = (R, t)
    
    if best_pose is None:
        return None, None
        
    return best_pose


