#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import requests
import tarfile
import tqdm
import yaml
from pathlib import Path
import numpy as np

# 模型配置
MODEL_CONFIG = {
    "small_model": {
        "url": "https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar",
        "params_dir": "params",
        "model_name": "params_model_1.npz"
    }
}

def download_file(url, output_path):
    """
    从指定URL下载文件到output_path
    """
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 下载文件
    print(f"正在下载 {url} 到 {output_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # 获取文件大小
    file_size = int(response.headers.get('content-length', 0))
    
    # 显示进度条并写入文件
    with open(output_path, 'wb') as f, tqdm.tqdm(
        desc=os.path.basename(output_path),
        total=file_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
                
    print(f"下载完成: {output_path}")
    return output_path

def extract_tarfile(tar_path, extract_dir):
    """
    解压tar文件到指定目录
    """
    print(f"正在解压 {tar_path} 到 {extract_dir}...")
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_dir)
    print(f"解压完成: {extract_dir}")

def create_mock_model_files(data_dir):
    """创建模拟的模型文件"""
    print("创建模拟模型文件...")
    
    # 创建模型目录
    model_dir = os.path.join(data_dir, "alphafold_params")
    os.makedirs(model_dir, exist_ok=True)
    
    # 创建一个模拟的模型文件
    model_file = os.path.join(model_dir, "params_model_1.npz")
    if not os.path.exists(model_file):
        np.savez(model_file, 
                 weights=np.random.rand(100, 100),
                 bias=np.random.rand(100))
    
    print(f"模拟模型文件已创建: {model_dir}")
    return model_dir

def download_model(model_name="small_model", data_dir="../data", use_mock=True):
    """
    下载并解压AlphaFold模型
    
    Args:
        model_name: 模型名称
        data_dir: 数据目录
        use_mock: 是否使用模拟模式（避免下载大文件）
    """
    try:
        if model_name not in MODEL_CONFIG:
            raise ValueError(f"不支持的模型: {model_name}, 可用模型: {list(MODEL_CONFIG.keys())}")
        
        config = MODEL_CONFIG[model_name]
        data_dir = os.path.abspath(data_dir)
        os.makedirs(data_dir, exist_ok=True)
        
        # 使用模拟模式
        if use_mock:
            extract_dir = create_mock_model_files(data_dir)
        else:
            # 下载tar文件
            tar_filename = os.path.basename(config["url"])
            tar_path = os.path.join(data_dir, tar_filename)
            
            if not os.path.exists(tar_path):
                download_file(config["url"], tar_path)
            else:
                print(f"模型文件已存在: {tar_path}")
            
            # 解压文件
            extract_dir = os.path.join(data_dir, "alphafold_params")
            if not os.path.exists(extract_dir):
                print(f"正在解压 {tar_path} 到 {data_dir}...")
                extract_tarfile(tar_path, data_dir)
            else:
                print(f"模型已解压: {extract_dir}")
        
        # 创建配置文件
        config_path = os.path.join(data_dir, "model_config.yaml")
        model_config = {
            "model": {
                "name": model_name,
                "params_dir": extract_dir,
                "model_file": config["model_name"]
            },
            "inference": {
                "max_seq_length": 1500,
                "num_ensemble": 1,
                "num_recycle": 3,
                "model_preset": "monomer"
            },
            "data": {
                "output_dir": os.path.join(data_dir, "predictions"),
                "pdb_cache_dir": os.path.join(data_dir, "pdb_cache"),
                "msa_cache_dir": os.path.join(data_dir, "msa_cache")
            }
        }
        
        # 创建必要的目录
        for dir_path in [
            model_config["data"]["output_dir"],
            model_config["data"]["pdb_cache_dir"],
            model_config["data"]["msa_cache_dir"]
        ]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 保存配置文件
        with open(config_path, 'w') as f:
            yaml.dump(model_config, f, default_flow_style=False)
        
        print(f"模型配置已保存: {config_path}")
        return model_config
        
    except Exception as e:
        print(f"下载模型时出错: {str(e)}")
        raise RuntimeError(f"模型下载失败: {str(e)}")

if __name__ == "__main__":
    download_model() 