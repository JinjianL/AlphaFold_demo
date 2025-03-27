#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import subprocess

def check_dependencies():
    """检查依赖项是否已安装"""
    try:
        import numpy
        import tensorflow
        import matplotlib
        import biopython
        import flask
        print("所有核心依赖项已安装")
        return True
    except ImportError as e:
        print(f"缺少依赖项: {e}")
        return False

def install_dependencies():
    """安装依赖项"""
    print("正在安装依赖项...")
    requirements_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])
        print("依赖项安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"依赖项安装失败: {e}")
        return False

def run_demo(host='127.0.0.1', port=5000, debug=False):
    """运行AlphaFold演示应用"""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "run_demo.py")
    
    cmd = [sys.executable, script_path, "--host", host, "--port", str(port)]
    if debug:
        cmd.append("--debug")
    
    print(f"启动AlphaFold演示应用，访问 http://{host}:{port}")
    subprocess.call(cmd)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AlphaFold演示应用启动器")
    parser.add_argument('--install', action='store_true', help='安装依赖项')
    parser.add_argument('--host', default='127.0.0.1', help='服务器主机')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    
    args = parser.parse_args()
    
    # 检查并安装依赖项
    if args.install or not check_dependencies():
        if not install_dependencies():
            print("请手动安装依赖项: pip install -r requirements.txt")
            return
    
    # 运行演示
    run_demo(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main() 