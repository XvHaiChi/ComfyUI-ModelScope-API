#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys
import os

def install_package(package):
    """安装Python包"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError as e:
        print(f"安装 {package} 失败: {e}")
        return False

def check_package(package_name):
    """检查包是否已安装"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def main():
    print("=" * 60)
    print("ComfyUI-ModelScope-API 依赖安装工具")
    print("=" * 60)
    
    # 检查核心依赖
    core_deps = {
        'requests': 'requests',
        'PIL': 'pillow',
        'torch': 'torch',
        'numpy': 'numpy'
    }
    
    print("\n🔍 检查核心依赖...")
    missing_core = []
    for import_name, package_name in core_deps.items():
        if check_package(import_name):
            print(f"✅ {package_name} 已安装")
        else:
            print(f"❌ {package_name} 未安装")
            missing_core.append(package_name)
    
    # 检查文本 / 图生文功能依赖
    print("\n🔍 检查文本 / 图生文功能依赖...")
    vision_deps = {
        'openai': 'openai',
        'httpx': 'httpx[socks]',
        'socksio': 'socksio'
    }
    
    missing_vision = []
    for import_name, package_name in vision_deps.items():
        if check_package(import_name):
            print(f"✅ {package_name} 已安装")
        else:
            print(f"❌ {package_name} 未安装")
            missing_vision.append(package_name)
    
    # 安装缺失的依赖
    all_missing = missing_core + missing_vision
    
    if not all_missing:
        print("\n🎉 所有依赖都已安装！")
        return
    
    print(f"\n📦 需要安装 {len(all_missing)} 个依赖包:")
    for pkg in all_missing:
        print(f"  - {pkg}")
    
    response = input("\n是否现在安装这些依赖？(y/n): ").lower().strip()
    
    if response in ['y', 'yes', '是']:
        print("\n🚀 开始安装依赖...")
        success_count = 0
        
        for package in all_missing:
            print(f"\n📦 安装 {package}...")
            if install_package(package):
                print(f"✅ {package} 安装成功")
                success_count += 1
            else:
                print(f"❌ {package} 安装失败")
        
        print(f"\n📊 安装结果: {success_count}/{len(all_missing)} 个包安装成功")
        
        if success_count == len(all_missing):
            print("🎉 所有依赖安装完成！请重启ComfyUI。")
        else:
            print("⚠️ 部分依赖安装失败，请手动安装或检查网络连接。")
            print("\n手动安装命令:")
            for package in all_missing:
                print(f"  pip install {package}")
    else:
        print("\n取消安装。")
        print("\n手动安装命令:")
        for package in all_missing:
            print(f"  pip install {package}")

if __name__ == "__main__":
    main()