import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time
from scipy.ndimage import laplace

def solve_laplace_sor(nx, ny, plate_thickness, plate_separation, omega=1.9, max_iter=10000, tolerance=1e-6):
    """
    使用SOR迭代法求解有限厚平行板电容器的拉普拉斯方程
    
    Args:
        nx (int): x方向网格点数
        ny (int): y方向网格点数
        plate_thickness (int): 极板厚度（网格点数）
        plate_separation (int): 极板间距（网格点数）
        omega (float): 松弛因子（1.0 < omega < 2.0）
        max_iter (int): 最大迭代次数
        tolerance (float): 收敛容差
        
    Returns:
        tuple: (potential_grid, conductor_mask)
            - potential_grid: 电势分布网格
            - conductor_mask: 导体区域掩码
    """
    # 初始化电势网格和导体掩码
    U = np.zeros((ny, nx))
    conductor_mask = np.zeros((ny, nx), dtype=bool)
    
    # 定义导体区域（中心对称）
    conductor_width = nx // 2
    conductor_left = (nx - conductor_width) // 2
    conductor_right = conductor_left + conductor_width
    
    # 上极板 (+100V)
    y_upper_start = ny // 2 + plate_separation // 2
    y_upper_end = y_upper_start + plate_thickness
    conductor_mask[y_upper_start:y_upper_end, conductor_left:conductor_right] = True
    U[y_upper_start:y_upper_end, conductor_left:conductor_right] = 100.0
    
    # 下极板 (-100V)
    y_lower_end = ny // 2 - plate_separation // 2
    y_lower_start = y_lower_end - plate_thickness
    conductor_mask[y_lower_start:y_lower_end, conductor_left:conductor_right] = True
    U[y_lower_start:y_lower_end, conductor_left:conductor_right] = -100.0
    
    # 边界条件：仅上下边界接地，左右边界开放
    U[0, :] = 0.0       # 上边界接地
    U[-1, :] = 0.0      # 下边界接地
    # 左右边界不接地，保持开放
    
    # SOR迭代
    for iteration in range(max_iter):
        U_old = U.copy()
        max_error = 0.0
        
        # 更新内部点（跳过导体和边界）
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                if not conductor_mask[i, j]:
                    # SOR更新公式
                    U_new = 0.25 * (U[i+1, j] + U[i-1, j] + U[i, j+1] + U[i, j-1])
                    U[i, j] = (1 - omega) * U[i, j] + omega * U_new
                    
                    # 跟踪最大误差
                    error = abs(U[i, j] - U_old[i, j])
                    max_error = max(max_error, error)
        
        # 检查收敛
        if max_error < tolerance:
            print(f"Converged after {iteration + 1} iterations")
            break
    else:
        print(f"Warning: Reached maximum iterations ({max_iter})")
    
    return U, conductor_mask

def calculate_charge_density(potential_grid, dx, dy):
    """
    通过泊松方程计算电荷密度 ρ = -∇²U/(4π)
    
    Args:
        potential_grid (np.ndarray): 电势分布网格
        dx (float): x方向网格间距
        dy (float): y方向网格间距
        
    Returns:
        np.ndarray: 电荷密度分布
    """
    # 计算x和y方向的二阶导数
    laplacian_x = (np.roll(potential_grid, -1, axis=1) - 2*potential_grid + np.roll(potential_grid, 1, axis=1)) / (dx**2)
    laplacian_y = (np.roll(potential_grid, -1, axis=0) - 2*potential_grid + np.roll(potential_grid, 1, axis=0)) / (dy**2)
    laplacian_U = laplacian_x + laplacian_y
    
    # 泊松方程计算电荷密度
    rho = -laplacian_U / (4 * np.pi)
    return rho

def plot_results(potential, charge_density, x_coords, y_coords, conductor_mask):
    """
    可视化电势和电荷密度分布，包含导体区域标记
    
    Args:
        potential (np.ndarray): 电势分布
        charge_density (np.ndarray): 电荷密度分布
        x_coords (np.ndarray): x坐标数组
        y_coords (np.ndarray): y坐标数组
        conductor_mask (np.ndarray): 导体区域掩码
    """
    X, Y = np.meshgrid(x_coords, y_coords)
    fig = plt.figure(figsize=(18, 8))
    
    # 子图1：电势等高线图（2D）
    ax1 = fig.add_subplot(121)
    contour = ax1.contourf(X, Y, potential, levels=50, cmap='viridis')
    ax1.contour(X, Y, potential, levels=20, colors='k', alpha=0.5, linewidths=0.5)
    ax1.contourf(X, Y, conductor_mask.astype(float), alpha=0.3, cmap='gray')  # 绘制导体区域
    ax1.set_title('电势分布 (V) 与导体区域')
    ax1.set_xlabel('X位置')
    ax1.set_ylabel('Y位置')
    plt.colorbar(contour, ax=ax1)
    
    # 子图2：电荷密度分布（2D）
    ax2 = fig.add_subplot(122)
    im = ax2.imshow(charge_density, cmap='RdBu_r', interpolation='nearest', 
                   extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]])
    ax2.contourf(X, Y, conductor_mask.astype(float), alpha=0.3, cmap='gray')  # 绘制导体区域
    ax2.set_title('电荷密度分布')
    ax2.set_xlabel('X位置')
    ax2.set_ylabel('Y位置')
    plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 模拟参数
    nx, ny = 120, 100          # 网格尺寸
    plate_thickness = 10       # 极板厚度（网格点数）
    plate_separation = 40      # 极板间距（网格点数）
    omega = 1.9                # SOR松弛因子
    
    # 物理尺寸
    Lx, Ly = 1.0, 1.0          # 定义域大小
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    
    # 创建坐标数组
    x_coords = np.linspace(0, Lx, nx)
    y_coords = np.linspace(0, Ly, ny)
    
    print("求解有限厚平行板电容器...")
    print(f"网格尺寸: {nx} x {ny}")
    print(f"极板厚度: {plate_thickness} 网格点")
    print(f"极板间距: {plate_separation} 网格点")
    print(f"SOR松弛因子: {omega}")
    
    # 求解拉普拉斯方程
    start_time = time.time()
    potential, conductor_mask = solve_laplace_sor(
        nx, ny, plate_thickness, plate_separation, omega
    )
    solve_time = time.time() - start_time
    print(f"求解完成，用时 {solve_time:.2f} 秒")
    
    # 计算电荷密度
    charge_density = calculate_charge_density(potential, dx, dy)
    
    # 可视化结果
    plot_results(potential, charge_density, x_coords, y_coords, conductor_mask)
    
    # 打印统计信息
    print(f"\n电势统计:")
    print(f"  最小电势: {np.min(potential):.2f} V")
    print(f"  最大电势: {np.max(potential):.2f} V")
    print(f"  电势范围: {np.max(potential) - np.min(potential):.2f} V")
    
    print(f"\n电荷密度统计:")
    max_charge = np.max(np.abs(charge_density))
    print(f"  最大电荷密度: {max_charge:.6f}")
    
    # 计算导体表面电荷（使用导体掩码）
    conductor_charge = charge_density[conductor_mask]
    total_positive = np.sum(conductor_charge[conductor_charge > 0]) * dx * dy
    total_negative = np.sum(conductor_charge[conductor_charge < 0]) * dx * dy
    print(f"  导体表面正电荷总量: {total_positive:.6f}")
    print(f"  导体表面负电荷总量: {total_negative:.6f}")
    print(f"  电荷总量: {total_positive + total_negative:.10f}")  # 理想情况下应接近零
