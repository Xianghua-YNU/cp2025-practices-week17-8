import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
from scipy.ndimage import laplace

def solve_laplace_sor(nx, ny, plate_thickness, plate_separation, omega=1.9, max_iter=10000, tolerance=1e-6):
    """
    使用逐次超松弛(SOR)方法求解二维拉普拉斯方程
    
    参数：
        nx (int): x方向的网格点数
        ny (int): y方向的网格点数  
        plate_thickness (int): 导体板厚度(网格点数)
        plate_separation (int): 板间距离(网格点数)
        omega (float): 松弛因子(1.0 < omega < 2.0)
        max_iter (int): 最大迭代次数
        tolerance (float): 收敛容差
        
    返回：
        np.ndarray: 二维电势分布数组
    """
    # 初始化电势网格
    U = np.zeros((ny, nx))
    
    # 创建导体区域掩模
    conductor_mask = np.zeros((ny, nx), dtype=bool)
    
    # 定义导体区域
    # 上极板：+100V
    conductor_left = nx//4   # 导体板左边界
    conductor_right = nx//4*3  # 导体板右边界
    y_upper_start = ny // 2 + plate_separation // 2  # 上极板起始y位置
    y_upper_end = y_upper_start + plate_thickness    # 上极板结束y位置
    # 设置上极板区域
    conductor_mask[y_upper_start:y_upper_end, conductor_left:conductor_right] = True
    U[y_upper_start:y_upper_end, conductor_left:conductor_right] = 100.0
    
    # 下极板：-100V
    y_lower_end = ny // 2 - plate_separation // 2    # 下极板结束y位置
    y_lower_start = y_lower_end - plate_thickness   # 下极板起始y位置
    # 设置下极板区域
    conductor_mask[y_lower_start:y_lower_end, conductor_left:conductor_right] = True
    U[y_lower_start:y_lower_end, conductor_left:conductor_right] = -100.0
    
    # 边界条件：接地边界
    U[:, 0] = 0.0    # 左边界
    U[:, -1] = 0.0   # 右边界
    U[0, :] = 0.0    # 上边界
    U[-1, :] = 0.0   # 下边界
    
    # SOR迭代
    for iteration in range(max_iter):
        U_old = U.copy()  # 保存前一次迭代结果
        max_error = 0.0   # 最大误差
        
        # 更新内部点(排除导体和边界)
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                if not conductor_mask[i, j]:  # 跳过导体点
                    # SOR更新公式
                    U_new = 0.25 * (U[i+1, j] + U[i-1, j] + U[i, j+1] + U[i, j-1])
                    U[i, j] = (1 - omega) * U[i, j] + omega * U_new
                    
                    # 跟踪最大误差
                    error = abs(U[i, j] - U_old[i, j])
                    max_error = max(max_error, error)
        
        # 检查收敛
        if max_error < tolerance:
            print(f"在 {iteration + 1} 次迭代后收敛")
            break
    else:
        print(f"警告：达到最大迭代次数 ({max_iter})")
    
    return U

def calculate_charge_density(potential_grid, dx, dy):
    """
    使用泊松方程计算电荷密度：rho = -1/(4*pi) * nabla^2(U)
    
    参数：
        potential_grid (np.ndarray): 二维电势分布
        dx (float): x方向网格间距
        dy (float): y方向网格间距
        
    返回：
        np.ndarray: 二维电荷密度分布
    """
    # 使用scipy.ndimage.laplace计算拉普拉斯算子
    laplacian_U = laplace(potential_grid, mode='nearest') / (dx**2)  # 假设dx=dy
    
    # 根据泊松方程计算电荷密度
    rho = -laplacian_U / (4 * np.pi)
    
    return rho

def plot_results(potential, charge_density, x_coords, y_coords):
    """
    创建结果的可视化图形（保持图像注释为英文）
    
    参数：
        potential (np.ndarray): 二维电势分布
        charge_density (np.ndarray): 电荷密度分布
        x_coords (np.ndarray): x坐标数组
        y_coords (np.ndarray): y坐标数组
    """
    X, Y = np.meshgrid(x_coords, y_coords)

    fig = plt.figure(figsize=(15, 6))

    # Subplot 1: 3D Visualization of Potential 
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_wireframe(X, Y, potential, rstride=3, cstride=3, color='r')
    levels = np.linspace(potential.min(), potential.max(), 20)
    ax1.contour(X, Y, potential, zdir='z', offset=potential.min(), levels=levels)
    ax1.set_title('3D Potential Distribution')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_zlabel('Potential (V)')

    # Subplot 2: 3D Charge Density Distribution
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(X, Y, charge_density, cmap='RdBu_r', edgecolor='none')
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5, label='Charge Density (C/m²)')
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Y Position (m)')
    ax2.set_zlabel('Charge Density (C/m²)')
    ax2.set_title('3D Charge Density Distribution')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 仿真参数设置
    nx, ny = 120, 100  # 网格尺寸
    plate_thickness = 10  # 导体板厚度(网格点数)
    plate_separation = 40  # 板间距离(网格点数)
    omega = 1.9  # SOR松弛因子
    
    # 物理尺寸设置
    Lx, Ly = 1.0, 1.0  # 计算域尺寸(米)
    dx = Lx / (nx - 1)  # x方向网格间距
    dy = Ly / (ny - 1)  # y方向网格间距
    
    # 创建坐标数组
    x_coords = np.linspace(0, Lx, nx)
    y_coords = np.linspace(0, Ly, ny)
    
    # 打印仿真信息
    print("正在求解有限厚度平行板电容器...")
    print(f"网格尺寸: {nx} x {ny}")
    print(f"极板厚度: {plate_thickness} 个网格点")
    print(f"极板间距: {plate_separation} 个网格点")
    print(f"SOR松弛因子: {omega}")
    
    # 求解拉普拉斯方程
    start_time = time.time()
    potential = solve_laplace_sor(
        nx, ny, plate_thickness, plate_separation, omega
    )
    solve_time = time.time() - start_time
    
    print(f"求解完成，耗时 {solve_time:.2f} 秒")
    
    # 计算电荷密度
    charge_density = calculate_charge_density(potential, dx, dy)
    
    # 可视化结果（图像注释保持英文）
    plot_results(potential, charge_density, x_coords, y_coords)
    
    # 打印统计信息
    print(f"\n电势统计:")
    print(f"  最小电势: {np.min(potential):.2f} V")
    print(f"  最大电势: {np.max(potential):.2f} V")
    print(f"  电势范围: {np.max(potential) - np.min(potential):.2f} V")
    
    print(f"\n电荷密度统计:")
    print(f"  最大电荷密度: {np.max(np.abs(charge_density)):.6f} C/m²")
    print(f"  总正电荷: {np.sum(charge_density[charge_density > 0]) * dx * dy:.6f} C")
    print(f"  总负电荷: {np.sum(charge_density[charge_density < 0]) * dx * dy:.6f} C")
    print(f"  总电荷: {np.sum(charge_density) * dx * dy:.6f} C")
