import sqlite3





def main():
    # 创建一个空的数据库
    conn = sqlite3.connect('Finance.db')
    #断开
    conn.close()

if __name__ == '__main__':
    main()
