import matplotlib.pyplot as plt

def visualize_point_cloud(point_cloud):
    if not point_cloud:
        print("Пустая точка облака")
        return
    
    X = [P[0][0] for P in point_cloud]
    Y = [P[1][0] for P in point_cloud]
    Z = [P[2][0] for P in point_cloud]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, s=1)
    plt.show()