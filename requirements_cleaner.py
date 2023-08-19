import subprocess

# 读取requirements.txt文件，明确指定编码方式
with open('requirements.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 将无特定路径的依赖项和特定路径的依赖项分开
no_path_lines = [line.strip() for line in lines if "file://" not in line]
path_lines = [line.strip().split("@")[0] for line in lines if "file://" in line]

# 存储最终的依赖项
final_requirements = no_path_lines.copy()

# 在命令行中获取每个包的实际版本
for i,package in enumerate(path_lines):
    package_name = package.split("==")[0]
    version_cmd = f"pip show {package_name}"
    print(f"正在执行第 {i}/{len(path_lines)} 个命令: {version_cmd}")
    version_output = subprocess.getoutput(version_cmd)

    # 查找版本行
    version_line = [line for line in version_output.split('\n') if "Version:" in line]
    if version_line:
        version = version_line[0].split(": ")[1]
        print(f"找到第 {i}/{len(path_lines)} 个包 {package_name} 的版本: {version}")
        final_requirements.append(f"{package_name}=={version}")
    else:
        print(f"无法找到包 {package_name} 的版本，命令输出: {version_output}")

# 将最终的依赖项写入新文件
with open('final_requirements.txt', 'w', encoding='utf-8') as file:
    file.write('\n'.join(final_requirements))

print("完成！最终的requirements文件已保存为final_requirements.txt")