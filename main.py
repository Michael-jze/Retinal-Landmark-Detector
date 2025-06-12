import cv2
import numpy as np
import math
import argparse

def tmp_plot(img, title, mask=None):
    """
    显示图像，可选叠加掩码
    img: 输入图像
    title: 窗口标题
    mask: 可选的掩码图像，将与原图叠加显示
    """
    img = img.copy()
    if mask is not None:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        img = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
    cv2.imshow(title, img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_image(image_path):
    """
    加载眼底图像
    image_path: 图像文件路径
    返回: 加载的图像
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("无法读取图像，请检查路径")
    return img

def calc_valid_mask(img):
    """
    计算有效区域掩码，排除图像边缘和背景区域
    img: 输入图像
    返回: 有效区域掩码 (255=有效, 0=无效)
    """
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 基于均值和标准差计算阈值
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    threshold = max(10, int(mean_val - std_val))
    
    # 创建掩码，低于阈值的区域标记为无效
    valid_mask = np.ones_like(gray) * 255
    valid_mask[gray < threshold] = 0
    
    print(f"valid_mask 信息 - 尺寸: {valid_mask.shape}, 均值: {np.mean(valid_mask)}")
    return valid_mask

def detect_disc(img, hsv, valid_mask):
    """
    检测视盘区域
    img: 原始图像
    hsv: HSV颜色空间的图像
    valid_mask: 有效区域掩码
    返回: 视盘区域掩码
    """
    # 定义视盘颜色范围（HSV空间）
    lower_disc = np.array([15, 80, 120])
    upper_disc = np.array([40, 255, 255])
    
    # 颜色阈值分割
    disc_mask = cv2.inRange(hsv, lower_disc, upper_disc)
    
    # 应用有效区域掩码
    disc_mask = cv2.bitwise_and(disc_mask, valid_mask)
    
    # 高斯模糊平滑边缘
    disc_mask = cv2.GaussianBlur(disc_mask, (5, 5), 0)
    
    # 形态学开运算去除小噪点
    kernel = np.ones((3, 3), np.uint8)
    disc_mask = cv2.morphologyEx(disc_mask, cv2.MORPH_OPEN, kernel)
    
    # 形态学闭运算填充小孔
    kernel = np.ones((7, 7), np.uint8)
    disc_mask = cv2.morphologyEx(disc_mask, cv2.MORPH_CLOSE, kernel)
    
    # 二值化处理
    disc_mask = cv2.threshold(disc_mask, 127, 255, cv2.THRESH_BINARY)[1]
    
    return disc_mask

def locate_disc_center_and_radius(disc_mask, img):
    """
    定位视盘中心和计算半径
    disc_mask: 视盘区域掩码
    img: 原始图像
    返回: 视盘中心坐标和半径
    """
    # 查找轮廓
    contours, _ = cv2.findContours(disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("警告：未检测到视盘")
        return None, None
    
    # 选择最大的轮廓作为视盘
    max_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(max_contour)
    
    # 过滤太小的区域
    if area < 500:
        print("警告：视盘区域太小")
        return None, None
    
    # 计算最小外接圆，获取中心和半径
    (center_x, center_y), radius = cv2.minEnclosingCircle(max_contour)
    center = (int(center_x), int(center_y))
    radius = int(radius)
    
    # 可视化视盘检测结果
    result = img.copy()
    cv2.circle(result, center, radius, (0, 255, 0), 2)
    cv2.putText(result, "Optic Disc", (center[0]+radius, center[1]-radius), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    tmp_plot(result, "Detected Optic Disc")
    
    return center, radius

def detect_vessels(img, valid_mask):
    """
    检测眼底图像中的血管结构
    img: 原始图像
    valid_mask: 有效区域掩码
    返回: 血管概率图和二值化的血管掩码
    """
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 应用有效区域掩码，确保只在有效区域内处理
    masked_gray = cv2.bitwise_and(gray, valid_mask)
    
    # 对比度增强，突出血管结构
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(masked_gray)
    
    # 高斯模糊去噪，减少伪边缘
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # 使用拉普拉斯算子进行边缘检测，特别适合检测线状结构（如血管）
    laplacian = cv2.Laplacian(blurred, cv2.CV_8U, ksize=3)
    
    # 阈值处理，增强血管结构
    _, thresh = cv2.threshold(laplacian, 10, 255, cv2.THRESH_BINARY)
    
    # 形态学操作优化血管结构：先膨胀连接断裂的血管，再腐蚀恢复原始大小
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    vessels_binary = cv2.erode(dilated, kernel, iterations=1)
    
    # 确保只在有效区域内
    vessels_binary = cv2.bitwise_and(vessels_binary, valid_mask)
    
    # 创建血管概率图（与二值化掩码相同，便于后续处理）
    vessels_prob = cv2.normalize(vessels_binary, None, 0, 255, cv2.NORM_MINMAX)
    
    return vessels_prob, vessels_binary

def detect_macula_based_on_disc(img, disc_center, disc_radius, valid_mask):
    """
    基于视盘位置检测黄斑
    img: 原始图像
    disc_center: 视盘中心坐标
    disc_radius: 视盘半径
    valid_mask: 有效区域掩码
    返回: 黄斑位置坐标
    """
    if disc_center is None or disc_radius is None:
        print("错误：未获取到视盘位置，无法检测黄斑")
        return None
    
    # 转换为RGB颜色空间（用于计算灰度值）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = gray.shape[:2]
    
    # 定义搜索区域：以视盘为中心，3倍视盘半径为半径的圆
    search_radius = min(disc_radius * 3, max(h, w) // 2)
    search_center = disc_center
    
    # 在圆周上均匀选取360个点作为候选位置（每1度一个点）
    num_candidates = 360
    macula_candidates = []
    
    # 检测血管
    vessels, vessels_binary = detect_vessels(img, valid_mask)
    tmp_plot(img, "Vessels Detection", vessels_binary)
    
    # 创建用于可视化候选点的图像
    candidate_visualization = img.copy()
    
    # 遍历所有候选点
    for i in range(num_candidates):
        # 计算圆周上的点
        angle = 2 * math.pi * i / num_candidates
        x = int(search_center[0] + search_radius * math.cos(angle))
        y = int(search_center[1] + search_radius * math.sin(angle))
        
        # 确保候选点在图像范围内
        if 0 <= x < w and 0 <= y < h:
            # 创建候选区域掩码：以当前点为中心，视盘半径为半径的圆
            candidate_mask = np.zeros_like(gray[:, :, 0])
            cv2.circle(candidate_mask, (x, y), disc_radius, 255, -1)
            
            # 应用有效区域掩码
            masked_candidate = cv2.bitwise_and(candidate_mask, valid_mask)
            
            # 计算候选区域总像素数
            total_pixels = np.sum(candidate_mask > 0)
            if total_pixels == 0:
                continue
                
            # 计算有效区域像素数
            valid_pixels = np.sum(masked_candidate > 0)
            valid_area_ratio = valid_pixels / total_pixels
            
            # 如果有效区域比例大于阈值，继续处理
            if valid_area_ratio >= 0.3:
                # 计算候选区域内的血管密度
                vessel_mask = cv2.bitwise_and(vessels_binary, masked_candidate)
                vessel_pixels = np.sum(vessel_mask > 0)
                vessel_density = vessel_pixels / valid_pixels if valid_pixels > 0 else 0
                
                # 计算候选区域的平均灰度值（值越小表示越暗）
                mean_gray = cv2.mean(gray, mask=masked_candidate)[0]
                
                # 记录候选点信息
                macula_candidates.append((x, y, mean_gray, valid_area_ratio, vessel_density))
                cv2.circle(candidate_visualization, (x, y), 3, (255, 0, 255), -1)
    
    # 显示所有候选点
    tmp_plot(candidate_visualization, "Macula Candidates")
    
    # 如果没有找到符合条件的候选点，返回None
    if not macula_candidates:
        print("警告：未找到符合条件的黄斑候选区域，尝试调整参数")
        return None
    
    # 按以下优先级排序：
    # 1. 血管密度最低（黄斑区域血管密度低）
    # 2. 灰度值最低（颜色最深）
    # 3. 有效区域比例最高
    macula_candidates.sort(key=lambda x: (x[4], x[2], -x[3]))
    best_candidate = macula_candidates[0]
    
    # 可视化最终结果
    visualization = img.copy()
    cv2.circle(visualization, search_center, search_radius, (255, 0, 0), 2)
    cv2.circle(visualization, (best_candidate[0], best_candidate[1]), disc_radius, (255, 0, 0), 2)
    cv2.putText(visualization, f"Macula (Vessel: {best_candidate[4]:.4f}, Gray: {best_candidate[2]:.1f})", 
               (best_candidate[0]+disc_radius, best_candidate[1]-disc_radius), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    tmp_plot(visualization, "Macula Localization Based on Disc and Vessels")
    
    return (best_candidate[0], best_candidate[1])

def detect_macula_disc(image_path):
    """
    主函数：检测眼底图像中的视盘和黄斑
    image_path: 眼底图像路径
    返回: 黄斑位置、视盘位置和视盘半径
    """
    # 加载图像
    img = load_image(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tmp_plot(img, "raw")
    
    # 计算有效区域掩码
    valid_mask = calc_valid_mask(img)
    tmp_plot(img, "valid_mask", valid_mask)
    
    # 转换为HSV颜色空间，用于视盘检测
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 检测视盘区域
    disc_mask = detect_disc(img, hsv, valid_mask)
    tmp_plot(img, "disc_mask", disc_mask)
    
    # 定位视盘中心和计算半径
    disc_center, disc_radius = locate_disc_center_and_radius(disc_mask, img)
    
    # 基于视盘位置检测黄斑
    macula_position = detect_macula_based_on_disc(img, disc_center, disc_radius, valid_mask)
    
    # 可视化最终结果
    result = img_rgb.copy()
    if disc_center and disc_radius:
        cv2.circle(result, disc_center, disc_radius, (0, 255, 0), 2)
        cv2.putText(result, "Optic Disc", (disc_center[0]+disc_radius, disc_center[1]-disc_radius), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if macula_position:
        cv2.circle(result, macula_position, 20, (255, 0, 0), 2)
        cv2.putText(result, "Macula", (macula_position[0]+25, macula_position[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    tmp_plot(result, "Final Result")
    
    return macula_position, disc_center, disc_radius

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="images/healthy.png")
    args = parser.parse_args()
    image_path = args.image_path
    
    # 执行视盘和黄斑检测
    macula_pos, disc_pos, disc_rad = detect_macula_disc(image_path)
    
    # 输出检测结果
    if macula_pos:
        print(f"黄斑位置: x={macula_pos[0]}, y={macula_pos[1]}")
    else:
        print("未能检测到黄斑")
    
    if disc_pos:
        print(f"视盘位置: x={disc_pos[0]}, y={disc_pos[1]}, 半径: {disc_rad}")
    else:
        print("未能检测到视盘")