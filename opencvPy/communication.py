import sys
import file_utils

def main(action):
    if action == "read":
        file_utils.read_and_print_file()
    elif action == "write":
        file_utils.write_data_to_file()
    else:
        print("无效的操作。使用 'read' 或 'write'。")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("用法: python communication.py <read|write>")
    else:
        action = sys.argv[1]
        main(action)
