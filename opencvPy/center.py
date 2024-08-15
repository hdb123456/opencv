import cv2
import json
import numpy as np

WHITE = (255, 255, 255)

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

def get_board_center(contour):
    """获取棋盘的中心点"""
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)
    return None

def draw_board_contour(image, contour):
    """在图像中绘制棋盘轮廓"""
    if contour is not None:
        cv2.drawContours(image, [contour], 0, WHITE, 2)
    return image

def draw_center(image, center):
    """在图像上绘制中心点"""
    if center is not None:
        cv2.circle(image, center, 5, WHITE, -1)
        cv2.putText(image, "Center", (center[0] - 20, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

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
            # 获取棋盘的中心点
            board_center = get_board_center(board_contour)

            # 绘制棋盘轮廓和中心点
            frame_with_board = frame.copy()
            frame_with_board = draw_board_contour(frame_with_board, board_contour)
            draw_center(frame_with_board, board_center)

            # 显示图像
            cv2.imshow('Center View', frame_with_board)

            if board_center is not None:
                # 输出中心点坐标
                print(f"{board_center[0]}\n{board_center[1]}")
                # 保存中心点坐标到文件
                with open('board_center.json', 'w') as f:
                    json.dump({"x": board_center[0], "y": board_center[1]}, f)
                break  # 中心点输出后退出循环

        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
