def matrix_multiply(A, B):
    """умножение матриц"""
    if len(A[0]) != len(B):
        return None
    
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

def matrix_transpose(A):
    """транспонирование матрицы"""
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def vector_normalize(v):
    """нормализация вектора"""
    norm = sum(x*x for x in v) ** 0.5
    return [x/norm for x in v] if norm > 0 else v

def power_iteration(A, num_iterations=100):
    """метод итерации по степеням"""
    #для вычисления максимального собственного значения и соответствующего собственного вектора
    n = len(A)
    # случайное инициализация вектора, после итерации сходится к собственному вектору с максимальным собственным значением
    v = [1.0/n**0.5 for _ in range(n)]
    
    # сохранение предыдущего собственного значения для проверки сходимости
    last_eigenvalue = 0
    
    for _ in range(num_iterations):
        # умножение матрицы на вектор
        Av = [sum(A[i][j]*v[j] for j in range(n)) for i in range(n)]
        # оценка собственного значения
        # Rayleigh Quotient = (v^T * A * v) / (v^T * v)
        # когда v - единичный вектор (v^T * v = 1)，Rayleigh Quotient упрощается: eigenvalue = v^T * A * v
        eigenvalue = sum(v[i]*Av[i] for i in range(n))
        
        # проверка сходимости
        if abs(eigenvalue - last_eigenvalue) < 1e-10:
            break
            
        last_eigenvalue = eigenvalue
        
        # нормализация
        norm = sum(x*x for x in Av) ** 0.5
        if norm > 1e-10:
            v = [x/norm for x in Av]
        else:
            return 0, [1 if i == 0 else 0 for i in range(n)]
    
    return eigenvalue, v

def deflate_matrix(A, eigenvalue, eigenvector):
    """Deflation матрицы"""
    #после нахождения собственного значения и собственного вектора находим новую матрицу для поиска остальных собственных значений
    #новая матрица = исходная матрица - λ(v * v^T)
    n = len(A)
    #вычисление внешнего произведения v * v^T
    outer_product = [[eigenvector[i]*eigenvector[j] for j in range(n)] 
                     for i in range(n)]
    #конструирование новой матрицы
    result = [[A[i][j] - eigenvalue*outer_product[i][j] 
              for j in range(n)] for i in range(n)]
    return result

#нормализация матрицы
def normalize_matrix(M):
    max_val = max(max(abs(x) for x in row) for row in M)
    if max_val > 1e-10:
        return [[x/max_val for x in row] for row in M]
    return M

 # обеспечение U и V являются ортогональными матрицами
def orthogonalize(M):
    # использование Gram-Schmidt метода для ортогонализации
    result = []
    for i in range(len(M)):
        v = M[i][:]
        # вычитание проекции предыдущих векторов
        for j in range(i):
            dot_product = sum(v[k]*M[j][k] for k in range(len(v)))
            v = [v[k] - dot_product*M[j][k] for k in range(len(v))]
        # нормализация
        norm = sum(x*x for x in v) ** 0.5
        if norm > 1e-10:
            v = [x/norm for x in v]
        else:
            v = [1 if j == i else 0 for j in range(len(v))]
        result.append(v)
    return result

def svd_solve(A, k=3):
    """SVD разложение"""
    """
    параметры:
    A: входная матрица
    k: количество вычисляемых сингулярных значений
    
    возвращает:
    U, S, V: результат SVD разложения
    """
    # обеспечкние того,что А - двумерный массив
    if not isinstance(A[0], list):
        A = [[x] for x in A]
    
    # вычисление A^T * A
    ATA = matrix_multiply(matrix_transpose(A), A)
    if ATA is None:
        return None, None, None
    
    n = len(ATA)
    
    # нормализация входной матрицы
    ATA = normalize_matrix(ATA)
    
    # инициализация результата
    V = []
    singular_values = []
    
    # итерация для вычисления первых k сингулярных значений и соответствующих правых сингулярных векторов
    current_matrix = ATA
    for i in range(k):
         # используем метод итерации по степеням для нахождения максимального собственного значения и соответствующего свободного вектора
        eigenvalue, eigenvector = power_iteration(current_matrix)
        if eigenvalue <= 1e-10:  
            # заполнение оставшихся сингулярных значений и векторов
            while len(singular_values) < k:
                singular_values.append(0)
                V.append([0] * n)
            break
            
        # вычисление правых сингулярных значений и добавление в список
        singular_value = abs(eigenvalue) ** 0.5
        singular_values.append(singular_value)
        V.append(eigenvector)
        
        # deflation матрицы, подготовка к поиску следующего
        current_matrix = deflate_matrix(current_matrix, eigenvalue, eigenvector)
        current_matrix = normalize_matrix(current_matrix)
    
    #обеспечение наличия достаточного количества векторов
    while len(V) < k:
        V.append([0] * n)
        singular_values.append(0)
    
    # вычисление левых сингулярных векторов
    U = []
    for i in range(k):
        if singular_values[i] > 1e-10:
            # u = Av/σ
            Av = matrix_multiply(A, [[x] for x in V[i]])
            if Av is None:
                u = [0] * len(A)
            else:
                u = [x[0]/singular_values[i] for x in Av]
                u = vector_normalize(u)
        else:
            u = [0] * len(A)
            u[i] = 1  # единичный вектор
        U.append(u)
    
    # конструирование диагональной матрицы S
    S = [[0 for _ in range(k)] for _ in range(k)]
    for i in range(k):
        S[i][i] = singular_values[i]
    
    U = orthogonalize(U)
    V = orthogonalize(V)
    
    return U, S, V

