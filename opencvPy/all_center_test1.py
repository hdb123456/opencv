import cv2
import json
import numpy as np
import subprocess
import time

WHITE = (255, 255, 255)
BLUE = (255, 0, 0)  # 用于绘制标记

def load_board_center():
    with open('board_center.json', 'r') as f:
        data = json.load(f)
        return (data["x"], data["y"])

KNOWN_WORLD_COORDINATE = (0, 0)
PIXEL_PER_CM = 45

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def find_board_contours(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def get_board_contour(contours):
    largest_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10000 and area > max_area:
            max_area = area
            largest_contour = contour
    return largest_contour

def get_board_center(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)
    return None

def get_square_centers(contour):
    squares = []
    x, y, w, h = cv2.boundingRect(contour)
    step_x = w // 3
    step_y = h // 3

    for i in range(3):
        for j in range(3):
            center_x = x + step_x * i + step_x // 2
            center_y = y + step_y * j + step_y // 2
            squares.append((center_x, center_y))

    return squares

def draw_board_contour(image, contour):
    if contour is not None:
        cv2.drawContours(image, [contour], 0, WHITE, 2)
    return image

def draw_center(image, center):
    if center is not None:
        cv2.circle(image, center, 5, WHITE, -1)
        cv2.putText(image, "Center", (center[0] - 20, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

def draw_square_centers(image, centers):
    square_coords = {}
    if centers:
        centers = sorted(centers, key=lambda c: (c[1], c[0]))
        num_columns = 3
        sorted_centers = []
        for i in range(0, len(centers), num_columns):
            sorted_centers.extend(sorted(centers[i:i + num_columns], key=lambda c: c[0]))

        for i, center in enumerate(sorted_centers):
            center = tuple(map(int, center))
            cv2.circle(image, center, 5, BLUE, -1)
            cv2.putText(image, str(i + 1), (center[0] + 10, center[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 2)

            world_coords = pixel_to_world(center[0], center[1])
            text = f"({world_coords[0]:.2f}, {world_coords[1]:.2f})"
            cv2.putText(image, text, (center[0] + 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1)

            # 保存坐标到字典
            square_coords[str(i + 1)] = {"x": world_coords[0], "y": world_coords[1]}

    # 将坐标保存到 JSON 文件中
    with open('square_centers.json', 'w') as f:
        json.dump(square_coords, f, indent=4)

    return square_coords

def pixel_to_world(pixel_x, pixel_y):
    delta_x = pixel_x - KNOWN_PIXEL_COORDINATE[0]
    delta_y = pixel_y - KNOWN_PIXEL_COORDINATE[1]
    world_x = delta_x / PIXEL_PER_CM
    world_y = delta_y / PIXEL_PER_CM
    return (world_x, world_y)

def run_center_script():
    result = subprocess.run(['python', 'center.py'], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"运行 center.py 失败: {result.stderr}")
        return False
    return True

def main():
    if not run_center_script():
        print("无法运行 center.py")
        return

    time.sleep(2)  # 等待 center.py 完成生成文件

    try:
        global KNOWN_PIXEL_COORDINATE
        KNOWN_PIXEL_COORDINATE = load_board_center()
    except FileNotFoundError:
        print("文件 board_center.json 不存在")
        return

    cap = cv2.VideoCapture(4)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    start_time = time.time()
    detected = False

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("读取摄像头帧失败或帧无效")
            break

        edges = preprocess_image(frame)
        contours = find_board_contours(edges)
        board_contour = get_board_contour(contours)

        if board_contour is not None:
            board_center = get_board_center(board_contour)
            square_centers = get_square_centers(board_contour)

            frame_with_board = frame.copy()
            frame_with_board = draw_board_contour(frame_with_board, board_contour)
            draw_center(frame_with_board, board_center)
            square_coords = draw_square_centers(frame_with_board, square_centers)

            if not detected:
                # 只在第一次检测到方格时打印坐标
                print("Detected square centers and world coordinates:")
                for i in range(1, 10):
                    coords = square_coords.get(str(i), {"x": 0, "y": 0})
                    print(f"Square {i}: Pixel ({square_centers[i-1][0]}, {square_centers[i-1][1]}) -> World ({coords['x']:.2f}, {coords['y']:.2f})")
                detected = True

            if time.time() - start_time >= 1:
                # 在 1 秒后显示最后的图像
                last_frame = frame_with_board
                cv2.imshow('Board Detection', last_frame)

                # 保存数据并退出程序
                cv2.imwrite('last_frame.png', last_frame)  # 保存最后一帧图像以供检查
                break  # 退出循环

        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
