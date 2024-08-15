import cv2
import numpy as np

# 棋盘和棋子相关参数
BOARD_SIZE = 3
CELL_SIZE = 100
BOARD_OFFSET = 50
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
EMPTY = 0
PLAYER_X = 1  # 黑色棋子
PLAYER_O = -1  # 白色棋子

# 霍夫圆变换参数
HOUGH_DP = 1.3
HOUGH_MIN_DIST = 30
HOUGH_PARAM1 = 50
HOUGH_PARAM2 = 30
HOUGH_MIN_RADIUS = 36
HOUGH_MAX_RADIUS = 56

# HSV颜色空间阈值
lower_white = np.array([0, 0, 150])
upper_white = np.array([180, 60, 255])

def put_piece(position, piece_type):
    row, col = position
    piece = "黑棋" if piece_type == 1 else "白棋"
    print(f"放置 {piece} 到位置 ({row}, {col})")

def preprocess_image(image):
    """预处理图像，增强对比度并转换为灰度图进行二值化"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(equalized, (9, 9), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    binary = cv2.medianBlur(binary, 3)
    return binary

def detect_circles(binary, dp, minDist, param1, param2, minRadius, maxRadius):
    """使用霍夫变换检测图像中的圆形棋子"""
    circles = cv2.HoughCircles(
        binary,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        print(f"检测到 {len(circles)} 个圆形")
        # 去重处理
        unique_circles = []
        for (x, y, r) in circles:
            if not any(np.sqrt((x - ux)**2 + (y - uy)**2) < r for (ux, uy, _) in unique_circles):
                unique_circles.append((x, y, r))
                print(f"圆形位置: ({x}, {y}), 半径: {r}")
        circles = np.array(unique_circles)
    else:
        print("未检测到圆形")
        circles = []
    return circles

def detect_board_contours(binary):
    """检测棋盘轮廓"""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def draw_board_contours(image, contours):
    """在图像中绘制棋盘轮廓"""
    for contour in contours:
        if cv2.contourArea(contour) > 10000:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.drawContours(image, [box], 0, WHITE, 2)
            break
    return image

def get_cell_position(x, y, cell_size, board_offset):
    """将像素坐标转换为棋盘格子坐标"""
    col = (x - board_offset) // cell_size
    row = (y - board_offset) // cell_size
    # 确保坐标在有效范围内
    row = max(0, min(BOARD_SIZE - 1, row))
    col = max(0, min(BOARD_SIZE - 1, col))
    print(f"像素坐标: ({x}, {y}) -> 棋盘坐标: ({row}, {col})")
    return row, col

def detect_colors(image):
    """使用HSV颜色空间检测黑白棋子"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_black = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    return mask_black, mask_white

def update_board_with_circles(board, circles, image, mask_black, mask_white):
    """根据检测到的圆形棋子更新棋盘，并在图像上绘制棋子"""
    black_pieces = set()  # 使用集合避免重复
    white_pieces = set()  # 使用集合避免重复

    if circles is not None:
        for (x, y, r) in circles:
            # 只处理像素坐标在有效范围内的棋子
            if 10 <= x <= 450 and 10 <= y <= 450:
                row, col = get_cell_position(x, y, CELL_SIZE, BOARD_OFFSET)
                # 确保位置在棋盘内
                if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                    # 检查棋子的颜色
                    color_detected = False
                    if mask_black[y, x] > 0:  # 黑色棋子
                        board[row, col] = PLAYER_X
                        cv2.circle(image, (x, y), r, BLACK, 2)
                        black_pieces.add((row, col))
                        color_detected = True
                    elif mask_white[y, x] > 0:  # 白色棋子
                        board[row, col] = PLAYER_O
                        cv2.circle(image, (x, y), r, WHITE, 2)
                        white_pieces.add((row, col))
                        color_detected = True

                    if color_detected:
                        # 打印棋子的坐标
                        print(f"检测到棋子位置: ({row}, {col})")
                        # 在图像上显示坐标
                        cv2.putText(image, f"({row}, {col})", (x - r, y - r - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)
                else:
                    print(f"位置 ({x}, {y}) 超出了棋盘范围")

    # 打印黑色和白色棋子的坐标
    if black_pieces:
        print("黑色棋子位置:")
        for piece in black_pieces:
            print(f"({piece[0]}, {piece[1]})")

    if white_pieces:
        print("白色棋子位置:")
        for piece in white_pieces:
            print(f"({piece[0]}, {piece[1]})")

    return board, black_pieces, white_pieces

def detect_pieces(frame, board, dp, minDist, param1, param2, minRadius, maxRadius):
    """在图像中检测棋盘和棋子"""
    binary = preprocess_image(frame)
    contours = detect_board_contours(binary)
    frame_with_contours = draw_board_contours(frame.copy(), contours)

    # 检测黑白棋子的掩膜
    mask_black, mask_white = detect_colors(frame)

    # 检测棋子
    circles = detect_circles(binary, dp, minDist, param1, param2, minRadius, maxRadius)

    # 在图像中只绘制检测到的黑色和白色圆形
    if circles is not None:
        for (x, y, r) in circles:
            if mask_black[y, x] > 0:
                cv2.circle(frame_with_contours, (x, y), r, BLACK, 2)  # 绘制黑色圆形
            elif mask_white[y, x] > 0:
                cv2.circle(frame_with_contours, (x, y), r, WHITE, 2)  # 绘制白色圆形

    board, black_pieces, white_pieces = update_board_with_circles(board, circles, frame_with_contours, mask_black, mask_white)

    return board, frame_with_contours, black_pieces, white_pieces


def get_piece_locations(frame):
    if frame is None or frame.size == 0:
        raise ValueError("提供的图像帧为空或无效。")

    """获取棋盘上棋子的位置信息"""
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

    dp = HOUGH_DP
    minDist = HOUGH_MIN_DIST
    param1 = HOUGH_PARAM1
    param2 = HOUGH_PARAM2
    minRadius = HOUGH_MIN_RADIUS
    maxRadius = HOUGH_MAX_RADIUS

    board, frame_with_pieces, black_pieces, white_pieces = detect_pieces(frame, board, dp, minDist, param1, param2, minRadius, maxRadius)

    return black_pieces, white_pieces, frame_with_pieces

def main():
    cap = cv2.VideoCapture(4)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            break

        black_pieces, white_pieces, frame_with_pieces = get_piece_locations(frame)
        print("黑棋位置:", black_pieces)
        print("白棋位置:", white_pieces)

        # 显示处理后的图像
        cv2.imshow("frame_with_pieces", frame_with_pieces)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
