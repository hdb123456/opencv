import json
import time
import cv2

from location import get_piece_locations, put_piece

# 棋盘和棋子相关参数
BOARD_SIZE = 3
EMPTY = 0
PLAYER_X = 1  # 黑色棋子
PLAYER_O = -1  # 白色棋子
WAIT_TIME_AFTER_MOVE = 10  # 每次放置后等待10秒
MOVE_TIME_LIMIT = 10  # 每方回合时间限制为10秒
MAX_MOVE_TIME = 15  # 最大落子时间限制为15秒

# 读取格子坐标和标号的 JSON 文件
with open('square_centers.json', 'r') as f:
    cell_data = json.load(f)

# 格子的标号和世界坐标映射
cell_map = {int(k): (v['x'], v['y']) for k, v in cell_data.items()}

global original_positions
original_positions = {}


def initialize_camera(camera_index=4):
    """初始化摄像头并返回摄像头对象"""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头。请检查摄像头连接。")
    return cap


def pixel_to_world(cell_number):
    """将格子标号转换为世界坐标"""
    if cell_number not in cell_map:
        raise ValueError(f"无效的格子标号: {cell_number}")

    # 返回格子的世界坐标
    return cell_map[cell_number]


def capture_frame(cap):
    """从摄像头捕获一帧图像"""
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        raise ValueError("捕获的图像帧为空或无效。")
    return frame


def find_empty_cells(board):
    """找到所有空白的格子位置"""
    empty_cells = []
    for row in range(len(board)):
        for col in range(len(board[row])):
            if board[row][col] == EMPTY:
                empty_cells.append((row, col))
    return empty_cells


def check_winner(board, player):
    """检查指定玩家是否赢得游戏"""
    # 检查行
    for row in range(BOARD_SIZE):
        if all(board[row][col] == player for col in range(BOARD_SIZE)):
            return True

    # 检查列
    for col in range(BOARD_SIZE):
        if all(board[row][col] == player for row in range(BOARD_SIZE)):
            return True

    # 检查对角线
    if all(board[i][i] == player for i in range(BOARD_SIZE)):
        return True
    if all(board[i][BOARD_SIZE - 1 - i] == player for i in range(BOARD_SIZE)):
        return True

    return False


def can_win(board, player):
    """检查是否有可以赢得游戏的位置"""
    empty_cells = find_empty_cells(board)
    for (row, col) in empty_cells:
        board[row][col] = player
        if check_winner(board, player):
            board[row][col] = EMPTY
            return (row, col)
        board[row][col] = EMPTY
    return None


def simple_ai_move(board):
    """这里我使用简单的策略来选择落子位置"""
    move = can_win(board, PLAYER_O)
    if move:
        print(f"AI选择的位置 ({move[0] * BOARD_SIZE + move[1] + 1}) 可以赢得游戏")
        return move
    move = can_win(board, PLAYER_X)
    if move:
        print(f"AI选择的位置 ({move[0] * BOARD_SIZE + move[1] + 1}) 可以阻止对手赢得游戏")
        return move

    # 如果没有即时赢或阻挡的机会，选择第一个空白的格子
    empty_cells = find_empty_cells(board)
    if empty_cells:
        next_move = empty_cells[0]
        print(f"AI选择的位置: ({next_move[0] * BOARD_SIZE + next_move[1] + 1})")
        return next_move

    return None


def ai_move(board, ai_piece):
    """AI进行一次移动"""
    # 检查中心位置 (5号格) 是否为空
    center_cell_number = 5
    center_cell_pos = ((center_cell_number - 1) // BOARD_SIZE, (center_cell_number - 1) % BOARD_SIZE)

    if board[center_cell_pos[0]][center_cell_pos[1]] == EMPTY:
        row, col = center_cell_pos
        board[row][col] = ai_piece
        print(f"AI在位置 ({center_cell_number}) 放置棋子")

        # 获取落子位置的世界坐标
        world_x, world_y = pixel_to_world(center_cell_number)
        print(f"AI准备落子的世界坐标: ({world_x:.2f}, {world_y:.2f})")

        put_piece((row, col), ai_piece)
        return (center_cell_number, (world_x, world_y))  # 返回 AI 落子位置和世界坐标

    # 如果中心位置被占用，使用简单策略选择其他位置
    move = simple_ai_move(board)
    if move:
        row, col = move
        board[row][col] = ai_piece
        print(f"AI在位置 ({row * BOARD_SIZE + col + 1}) 放置棋子")

        # 获取落子位置的世界坐标
        cell_number = row * BOARD_SIZE + col + 1
        world_x, world_y = pixel_to_world(cell_number)
        print(f"AI准备落子的世界坐标: ({world_x:.2f}, {world_y:.2f})")

        put_piece((row, col), ai_piece)
        return (cell_number, (world_x, world_y))  # 返回 AI 落子位置和世界坐标
    else:
        print("AI无法选择位置")
        return None


def player_move(board, player_piece, cap):
    """处理玩家的移动"""
    if cap is None:
        print("错误：摄像头未初始化。")
        return

    last_piece_count = 0
    detected = False

    start_time = time.time()  # 记录开始时间

    while not detected:
        try:
            frame = capture_frame(cap)

            # 从当前帧中检测棋子
            black_pieces, white_pieces, frame_with_pieces = get_piece_locations(frame)
            print(f"检测到的黑色棋子位置: {black_pieces}")
            print(f"检测到的白色棋子位置: {white_pieces}")

            # 计算当前棋子的总数量
            current_piece_count = len(black_pieces) + len(white_pieces)

            # 如果检测到棋子数目增加，更新棋盘状态
            if current_piece_count > last_piece_count:
                for (r, c) in black_pieces:
                    board[r][c] = PLAYER_X
                for (r, c) in white_pieces:
                    board[r][c] = PLAYER_O
                last_piece_count = current_piece_count
                detected = True
            else:
                # 如果没有检测到新的棋子，等待1秒再轮询
                time.sleep(1)

            # 处理窗口关闭事件
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("用户退出了程序。")
                break

            # 检查是否超时
            elapsed_time = time.time() - start_time
            if elapsed_time > MAX_MOVE_TIME:
                print("玩家落子超时，回合结束。")
                break

            if detected:
                # 处理玩家落子时间限制
                elapsed_time = time.time() - start_time
                if elapsed_time > MOVE_TIME_LIMIT:
                    remaining_time = int(MOVE_TIME_LIMIT - elapsed_time)
                    print(f"玩家落子时间结束，剩余时间：{remaining_time}秒。请按空格键继续。")
                    cv2.waitKey(-1)  # 等待空格键被按下

        except ValueError as ve:
            print(f"值错误: {ve}")
            break
        except Exception as e:
            print(f"玩家移动检测时发生错误: {e}")
            break

    # 检查棋子是否移动并恢复丢失的棋子
    detect_and_restore_pieces(board)

    print("玩家移动处理结束")
    print_board(board)  # 打印当前棋盘状态


def detect_and_restore_pieces(board):
    """检测并恢复棋子位置"""
    global original_positions

    try:
        # 获取当前棋子的位置
        current_positions = {
            (r, c): board[r][c]
            for r in range(BOARD_SIZE)
            for c in range(BOARD_SIZE)
            if board[r][c] != EMPTY
        }

        # 遍历原始位置，检查是否有棋子丢失
        for (r, c), piece in original_positions.items():
            if (r, c) not in current_positions:
                # 棋子丢失，恢复原来的棋子
                board[r][c] = piece
                put_piece((r, c), piece)  # 恢复棋子位置
                print(f"棋子被移动，恢复棋子到位置 ({r * BOARD_SIZE + c + 1})")

        # 更新原始位置为当前棋子的位置
        original_positions = current_positions

    except Exception as e:
        print(f"检测和恢复棋子位置时发生错误: {e}")


def choose_pieces():
    """选择棋子"""
    while True:
        choice = input("请选择棋子（输入 '1' 选择黑棋，输入 '0' 选择白棋）：")
        if choice == '1':
            return (PLAYER_X, PLAYER_O, "玩家")
        elif choice == '0':
            return (PLAYER_O, PLAYER_X, "玩家")
        else:
            print("无效选择，请重新输入")


def print_board(board):
    """打印棋盘"""
    print("\n棋盘状态:")
    for row in board:
        print(' '.join(str(cell) if cell != EMPTY else '.' for cell in row))
    print()


def play_game():
    """游戏主循环"""
    cap = initialize_camera()  # 初始化摄像头

    board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    player_piece, ai_piece, first_turn = choose_pieces()
    current_player = first_turn

    # 记录初始棋子位置
    global original_positions
    original_positions = {
        (r, c): EMPTY
        for r in range(BOARD_SIZE)
        for c in range(BOARD_SIZE)
    }

    print_board(board)

    while True:
        if current_player == "玩家":
            print("玩家的回合")
            player_move(board, player_piece, cap)
            current_player = "AI"
        else:
            print("AI的回合")
            ai_move(board, ai_piece)
            current_player = "玩家"

        print_board(board)

        # 检查是否有赢家
        if check_winner(board, player_piece):
            print("玩家获胜！")
            break
        elif check_winner(board, ai_piece):
            print("AI获胜！")
            break

        # 检查是否平局
        if all(cell != EMPTY for row in board for cell in row):
            print("平局！")
            break

        # 每次移动后等待一定时间
        time.sleep(WAIT_TIME_AFTER_MOVE)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    play_game()
