import cv2
import numpy as np

WHITE = (255, 255, 255)
selected_points = []

def preprocess_image(image):
    """预处理图像，使用边缘检测和二值化方法"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def find_board_contours(edges):
    """检测棋盘轮廓"""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_board_contour(contours):
    """选择最适合的棋盘轮廓"""
    for contour in contours:
        if cv2.contourArea(contour) > 10000:  # 只考虑大轮廓
            return contour
    return None

def draw_points(image, points):
    """在图像上绘制点"""
    for point in points:
        cv2.circle(image, tuple(map(int, point)), 10, WHITE, -1)
    return image

def click_event(event, x, y, flags, params):
    """鼠标点击事件处理"""
    global selected_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(selected_points) < 4:
            selected_points.append((x, y))
            print(f"选择的点: {(x, y)}")

def get_image_points():
    """获取图像中的四个角点坐标"""
    global selected_points
    if len(selected_points) == 4:
        return np.array(selected_points, dtype=float)
    else:
        raise ValueError("请选择四个角点")

def convert_to_world_coords(image_points, origin=(224, 233), pixels_per_cm=150):
    """将像素坐标转换为世界坐标（单位：厘米）"""
    world_coords = []
    for i, point in enumerate(image_points):
        # 计算相对坐标
        x = (point[0] - origin[0]) / pixels_per_cm
        y = (point[1] - origin[1]) / pixels_per_cm
        world_coords.append((x, y))
        print(f"点 {i + 1} 的世界坐标: ({x:.2f}, {y:.2f}) cm")
    return np.array(world_coords)

def calculate_midpoint(points):
    """计算四个点对角线的交点"""
    if len(points) == 4:
        def line_intersection(p1, p2, p3, p4):
            """计算两条线段的交点"""
            A1 = p2[1] - p1[1]
            B1 = p1[0] - p2[0]
            C1 = A1 * p1[0] + B1 * p1[1]

            A2 = p4[1] - p3[1]
            B2 = p3[0] - p4[0]
            C2 = A2 * p3[0] + B2 * p3[1]

            determinant = A1 * B2 - A2 * B1
            if determinant == 0:
                raise ValueError("线段平行，无交点")
            x = (B2 * C1 - B1 * C2) / determinant
            y = (A1 * C2 - A2 * C1) / determinant
            return (x, y)

        top_left = points[0]
        top_right = points[1]
        bottom_left = points[2]
        bottom_right = points[3]

        midpoints = [
            line_intersection(top_left, bottom_right, top_right, bottom_left)
        ]

        return np.array(midpoints)
    else:
        raise ValueError("需要四个点来计算中点")

def main():
    # 初始化摄像头
    cap = cv2.VideoCapture(4)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("读取摄像头帧失败或帧无效")
            break

        # 预处理图像
        edges = preprocess_image(frame)

        # 检测棋盘轮廓
        contours = find_board_contours(edges)
        board_contour = get_board_contour(contours)

        if board_contour is not None:
            # 绘制棋盘轮廓
            frame_with_board = frame.copy()
            frame_with_board = draw_points(frame_with_board, selected_points)

            # 显示图像
            cv2.imshow('Center View', frame_with_board)

            # 点击选择角点
            cv2.setMouseCallback('Center View', click_event)

        # 检测空格键
        if cv2.waitKey(1) & 0xFF == ord(' '):
            if len(selected_points) == 4:
                image_points = get_image_points()
                world_coords = convert_to_world_coords(image_points)
                print("世界坐标: ", world_coords)

                # 计算对角线交点坐标
                midpoints = calculate_midpoint(image_points)
                world_midpoint = convert_to_world_coords(midpoints, origin=(224, 233), pixels_per_cm=150)
                print("中点坐标: ", world_midpoint)
                break
            else:
                print("请先选择四个角点")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
