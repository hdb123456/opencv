import json
import sys

READ_FILENAME = 'read_file.json'
WRITE_FILENAME = 'write_file.json'

def read_and_print_file():
    """
    从文件中读取数据并打印到标准输出
    """
    try:
        with open(READ_FILENAME, 'r', encoding='utf-8') as file:
            data = file.read()
            print(data)
    except FileNotFoundError:
        print(f"文件 {READ_FILENAME} 未找到。")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")

def write_data_to_file():
    """
    从标准输入接收数据并写入文件
    """
    print("请输入数据（按 Ctrl+D 结束输入）：")
    try:
        # 读取标准输入数据
        input_data = sys.stdin.read()

        # 尝试解析输入数据为 JSON
        try:
            json_data = json.loads(input_data)
            formatted_json = json.dumps(json_data, indent=4, ensure_ascii=False)
        except json.JSONDecodeError:
            print("输入的数据不是有效的 JSON 格式，将原样写入文件。")
            formatted_json = input_data

        # 将数据写入文件
        with open(WRITE_FILENAME, 'w', encoding='utf-8') as file:
            file.write(formatted_json)

        print(f"数据已保存到 {WRITE_FILENAME}")
    except Exception as e:
        print(f"写入文件时发生错误: {e}")
