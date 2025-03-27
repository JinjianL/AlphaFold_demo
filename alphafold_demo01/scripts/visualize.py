#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import py3Dmol
from IPython.display import display
import seaborn as sns
from pathlib import Path

def plot_confidence(result, output_dir):
    """
    绘制预测置信度图
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制每个残基的pLDDT得分
    residue_indices = np.arange(1, len(result['plddt']) + 1)
    plt.plot(residue_indices, result['plddt'], '-', linewidth=2)
    
    # 添加区域颜色指示不同的置信度级别
    plt.axhspan(0, 50, alpha=0.1, color='red', label='Very low (pLDDT < 50)')
    plt.axhspan(50, 70, alpha=0.1, color='orange', label='Low (70 > pLDDT > 50)')
    plt.axhspan(70, 90, alpha=0.1, color='yellow', label='Confident (90 > pLDDT > 70)')
    plt.axhspan(90, 100, alpha=0.1, color='green', label='Very confident (pLDDT > 90)')
    
    plt.title('AlphaFold Prediction Confidence (pLDDT)')
    plt.xlabel('Residue Position')
    plt.ylabel('Predicted LDDT (pLDDT)')
    plt.ylim(0, 100)
    plt.xlim(1, len(result['plddt']))
    plt.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0))
    
    # 保存图形
    os.makedirs(output_dir, exist_ok=True)
    confidence_plot_path = os.path.join(output_dir, 'confidence_plot.png')
    plt.savefig(confidence_plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Confidence plot saved to: {confidence_plot_path}")
    return confidence_plot_path

def generate_pymol_script(pdb_path, output_dir):
    """
    生成PyMOL可视化脚本
    """
    pymol_script = f"""
# PyMOL脚本用于可视化蛋白质结构
load {pdb_path}
bg_color white
as cartoon
spectrum b, rainbow
zoom
ray 1200, 1200
save {os.path.join(output_dir, 'structure_render.png')}
    """
    
    script_path = os.path.join(output_dir, 'visualize.pml')
    with open(script_path, 'w') as f:
        f.write(pymol_script)
    
    print(f"PyMOL脚本已保存到: {script_path}")
    return script_path

def visualize_3d(pdb_path, width=800, height=500):
    """
    使用py3Dmol创建交互式3D可视化
    """
    try:
        # 读取PDB文件内容
        with open(pdb_path, 'r') as f:
            pdb_content = f.read()
        
        # 创建查看器
        viewer = py3Dmol.view(width=width, height=height)
        viewer.addModel(pdb_content, "pdb")
        viewer.setStyle({}, {'cartoon': {'color': 'spectrum'}})
        viewer.setBackgroundColor('white')
        viewer.zoomTo()
        
        # 返回HTML/JavaScript代码
        html = viewer._make_html()
        
        # 清理viewer以释放内存
        viewer = None
        
        return html
        
    except Exception as e:
        print(f"3D可视化错误: {str(e)}")
        return None

def visualize_results(result, pdb_path, output_dir):
    """
    综合可视化预测结果
    """
    # 绘制置信度图
    confidence_plot_path = plot_confidence(result, output_dir)
    
    # 生成PyMOL脚本
    pymol_script_path = generate_pymol_script(pdb_path, output_dir)
    
    # 创建结果报告
    report_path = os.path.join(output_dir, 'results_report.txt')
    with open(report_path, 'w') as f:
        f.write("AlphaFold预测结果报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"序列长度: {len(result['sequence'])}\n")
        f.write(f"平均pLDDT得分: {np.mean(result['plddt']):.2f}\n")
        f.write(f"最高pLDDT得分: {np.max(result['plddt']):.2f}\n")
        f.write(f"最低pLDDT得分: {np.min(result['plddt']):.2f}\n\n")
        f.write("文件路径:\n")
        f.write(f"- PDB结构文件: {pdb_path}\n")
        f.write(f"- 置信度图: {confidence_plot_path}\n")
        f.write(f"- PyMOL脚本: {pymol_script_path}\n")
    
    print(f"结果报告已保存到: {report_path}")
    
    # 返回可视化产物的路径
    visualization_results = {
        'pdb_path': pdb_path,
        'confidence_plot': confidence_plot_path,
        'pymol_script': pymol_script_path,
        'report': report_path
    }
    
    return visualization_results

if __name__ == "__main__":
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser(description="可视化AlphaFold预测结果")
    parser.add_argument('--pdb', required=True, help='PDB结构文件路径')
    parser.add_argument('--output', default='../output', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建伪结果数据（实际使用时会从预测结果中获取）
    sequence = "MAETGKYFTLADEALKWDFADLSEKLSDEGDLFRVGMFRPSVSEPIGERAVLITMEKGYEFKSRGIDVNHVDLDDVIEIYSKHTDGNSVDVVKLFIENGKGSPESIRKWIQKYAEFPTLQVAWCSVGNE"
    result = {
        'sequence': sequence,
        'plddt': np.random.rand(len(sequence)) * 70 + 30
    }
    
    visualize_results(result, args.pdb, args.output) 