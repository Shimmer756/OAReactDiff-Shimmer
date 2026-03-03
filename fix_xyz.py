def fix_xyz_format(input_file="generated_meci.xyz", output_file="generated_meci_fixed.xyz"):
    # 元素周期表映射字典 (支持常见的有机元素)
    atom_map = {"1": "H", "6": "C", "7": "N", "8": "O", "9": "F", "15": "P", "16": "S", "17": "Cl"}
    
    with open(input_file, "r") as f:
        lines = f.readlines()
        
    with open(output_file, "w") as f:
        for line in lines:
            parts = line.split()
            # 如果这一行有 4 个数据，且第一个数字在我们的字典里，就把它替换成字母
            if len(parts) == 4 and parts[0] in atom_map:
                symbol = atom_map[parts[0]]
                f.write(f"{symbol:<2} {parts[1]:>10} {parts[2]:>10} {parts[3]:>10}\n")
            else:
                f.write(line)
                
    print(f"✅ 转换成功！修复后的文件已保存为: {output_file}")

if __name__ == "__main__":
    fix_xyz_format()
