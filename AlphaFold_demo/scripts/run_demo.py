#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import tempfile
from pathlib import Path
import time
import base64
import io

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import flask
from flask import Flask, request, render_template, send_file, redirect, url_for, jsonify

from model import SimplifiedAlphaFold
from prepare_data import prepare_sequence, get_sample_sequence, SAMPLE_SEQUENCES
from visualize import visualize_results, plot_confidence, visualize_3d
from download_model import download_model

# 创建Flask应用
app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../templates'),
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../static'))

# 配置目录
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../output')

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 全局模型实例
model = None

def get_model():
    """获取或初始化模型"""
    global model
    
    if model is None:
        config_path = os.path.join(DATA_DIR, 'model_config.yaml')
        
        # 如果模型配置不存在，则下载模型
        if not os.path.exists(config_path):
            print("模型配置不存在，正在下载模型...")
            download_model(data_dir=DATA_DIR, use_mock=True)  # 使用模拟模式
        
        try:
            # 初始化模型
            model = SimplifiedAlphaFold(config_path)
        except Exception as e:
            print(f"模型初始化失败: {e}")
            # 如果失败，尝试重新创建配置
            download_model(data_dir=DATA_DIR, use_mock=True)
            model = SimplifiedAlphaFold(config_path)
    
    return model

def figure_to_base64(fig):
    """将matplotlib图形转换为base64编码的PNG图像"""
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return base64.b64encode(output.getvalue()).decode('utf-8')

@app.route('/')
def index():
    """主页"""
    # 获取最近的预测结果
    recent_results = []
    if os.path.exists(OUTPUT_DIR):
        for result_dir in sorted(os.listdir(OUTPUT_DIR), reverse=True)[:5]:
            result_path = os.path.join(OUTPUT_DIR, result_dir)
            if os.path.isdir(result_path):
                report_path = os.path.join(result_path, 'results_report.txt')
                if os.path.exists(report_path):
                    recent_results.append({
                        'name': result_dir,
                        'time': time.ctime(os.path.getctime(report_path)),
                        'path': result_path
                    })
    
    # 获取示例序列
    examples = list(SAMPLE_SEQUENCES.keys())
    
    return render_template('index.html', 
                           recent_results=recent_results,
                           examples=examples)

@app.route('/predict', methods=['POST'])
def predict():
    """处理预测请求"""
    # 获取输入序列
    sequence_input = request.form.get('sequence', '').strip()
    use_example = request.form.get('use_example', '')
    
    if not sequence_input and not use_example:
        return jsonify({'error': '请提供蛋白质序列或选择示例'}), 400
    
    if use_example:
        name, sequence = get_sample_sequence(use_example)
    else:
        name = request.form.get('name', f"sequence_{int(time.time())}")
        sequence = sequence_input
    
    # 创建输出目录
    timestamp = int(time.time())
    result_dir = f"{name}_{timestamp}"
    output_dir = os.path.join(OUTPUT_DIR, result_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备序列
    seq_data = prepare_sequence(sequence, DATA_DIR, name)
    if not seq_data:
        return jsonify({'error': '序列准备失败'}), 400
    
    # 获取模型并预测
    try:
        model = get_model()
        result, pdb_path = model.predict(seq_data['sequence'], output_dir)
        
        # 可视化结果
        vis_results = visualize_results(result, pdb_path, output_dir)
        
        # 重定向到结果页面
        return redirect(url_for('result', result_id=result_dir))
    
    except Exception as e:
        return jsonify({'error': f'预测失败: {str(e)}'}), 500

@app.route('/result/<result_id>')
def result(result_id):
    """显示预测结果"""
    result_dir = os.path.join(OUTPUT_DIR, result_id)
    
    if not os.path.exists(result_dir):
        return "结果不存在", 404
    
    # 读取报告
    report_path = os.path.join(result_dir, 'results_report.txt')
    report_content = ""
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            report_content = f.read()
    
    # 获取文件路径
    pdb_path = os.path.join(result_dir, 'predicted_structure.pdb')
    confidence_plot_path = os.path.join(result_dir, 'confidence_plot.png')
    
    # 检查文件是否存在
    files = {
        'pdb': os.path.exists(pdb_path),
        'confidence_plot': os.path.exists(confidence_plot_path)
    }
    
    return render_template('result.html',
                           result_id=result_id,
                           report=report_content,
                           files=files)

@app.route('/download/<result_id>/<file_type>')
def download(result_id, file_type):
    """下载结果文件"""
    result_dir = os.path.join(OUTPUT_DIR, result_id)
    
    if file_type == 'pdb':
        file_path = os.path.join(result_dir, 'predicted_structure.pdb')
        return send_file(file_path, as_attachment=True, download_name=f"{result_id}.pdb")
    
    elif file_type == 'confidence_plot':
        file_path = os.path.join(result_dir, 'confidence_plot.png')
        return send_file(file_path, as_attachment=True, download_name=f"{result_id}_confidence.png")
    
    elif file_type == 'report':
        file_path = os.path.join(result_dir, 'results_report.txt')
        return send_file(file_path, as_attachment=True, download_name=f"{result_id}_report.txt")
    
    return "文件不存在", 404

@app.route('/view_3d/<job_id>')
def view_3d(job_id):
    try:
        # 获取PDB文件路径
        output_dir = os.path.join(OUTPUT_DIR, job_id)
        pdb_path = os.path.join(output_dir, 'predicted_structure.pdb')
        
        if not os.path.exists(pdb_path):
            return "结构文件不存在", 404
            
        # 创建3D可视化
        viewer_code = visualize_3d(pdb_path)
        if viewer_code is None:
            return "3D可视化生成失败", 500
            
        # 返回包含3D查看器的HTML页面
        return render_template(
            'viewer_3d.html',
            result_id=job_id,
            viewer_code=viewer_code
        )
        
    except Exception as e:
        print(f"3D可视化错误: {str(e)}")
        return str(e), 500

@app.route('/get_pdb/<result_id>')
def get_pdb(result_id):
    """获取PDB文件内容"""
    result_dir = os.path.join(OUTPUT_DIR, result_id)
    pdb_path = os.path.join(result_dir, 'predicted_structure.pdb')
    
    if not os.path.exists(pdb_path):
        return "PDB文件不存在", 404
    
    with open(pdb_path, 'r') as f:
        pdb_content = f.read()
    
    return pdb_content

def create_templates():
    """创建HTML模板"""
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # 创建首页模板
    index_html = """<!DOCTYPE html>
<html>
<head>
    <title>AlphaFold演示</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css">
    <style>
        body {
            padding-top: 20px;
            padding-bottom: 20px;
        }
        .header {
            padding-bottom: 20px;
            border-bottom: 1px solid #e5e5e5;
            margin-bottom: 30px;
        }
        .protein-form {
            margin-bottom: 30px;
        }
        .recent-results {
            margin-top: 40px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>AlphaFold蛋白质结构预测演示</h1>
            <p class="lead">输入蛋白质序列，获取3D结构预测</p>
        </div>
        
        <div class="row">
            <div class="col-md-8">
                <div class="protein-form">
                    <h2>预测新结构</h2>
                    <form action="/predict" method="post">
                        <div class="mb-3">
                            <label for="name" class="form-label">名称 (可选)</label>
                            <input type="text" class="form-control" id="name" name="name" placeholder="为这个预测命名">
                        </div>
                        
                        <div class="mb-3">
                            <label for="sequence" class="form-label">蛋白质序列</label>
                            <textarea class="form-control" id="sequence" name="sequence" rows="6" placeholder="输入蛋白质氨基酸序列..."></textarea>
                        </div>
                        
                        <div class="mb-3">
                            <label class="form-label">或选择示例序列:</label>
                            <select class="form-select" id="example-select" name="use_example">
                                <option value="">-- 选择示例 --</option>
                                {% for example in examples %}
                                <option value="{{ example }}">{{ example }}</option>
                                {% endfor %}
                            </select>
                        </div>
                        
                        <button type="submit" class="btn btn-primary">预测结构</button>
                    </form>
                </div>
            </div>
            
            <div class="col-md-4">
                <div class="recent-results">
                    <h3>最近的预测</h3>
                    {% if recent_results %}
                    <ul class="list-group">
                        {% for result in recent_results %}
                        <li class="list-group-item">
                            <a href="/result/{{ result.name }}">{{ result.name }}</a>
                            <small class="text-muted d-block">{{ result.time }}</small>
                        </li>
                        {% endfor %}
                    </ul>
                    {% else %}
                    <p>还没有预测结果</p>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // 示例序列选择
        document.getElementById('example-select').addEventListener('change', function() {
            if (this.value) {
                document.getElementById('sequence').value = '';
            }
        });
        
        // 清空示例选择
        document.getElementById('sequence').addEventListener('input', function() {
            if (this.value) {
                document.getElementById('example-select').value = '';
            }
        });
    </script>
</body>
</html>
"""
    
    # 创建结果页面模板
    result_html = """<!DOCTYPE html>
<html>
<head>
    <title>AlphaFold预测结果</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css">
    <style>
        body {
            padding-top: 20px;
            padding-bottom: 20px;
        }
        .header {
            padding-bottom: 20px;
            border-bottom: 1px solid #e5e5e5;
            margin-bottom: 30px;
        }
        .result-section {
            margin-bottom: 30px;
        }
        pre {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>预测结果: {{ result_id }}</h1>
            <p><a href="/" class="btn btn-outline-secondary btn-sm">返回首页</a></p>
        </div>
        
        <div class="row">
            <div class="col-md-7">
                <div class="result-section">
                    <h3>3D结构</h3>
                    <p>
                        <a href="/view_3d/{{ result_id }}" class="btn btn-primary" target="_blank">查看3D结构</a>
                        {% if files.pdb %}
                        <a href="/download/{{ result_id }}/pdb" class="btn btn-outline-secondary">下载PDB文件</a>
                        {% endif %}
                    </p>
                </div>
                
                <div class="result-section">
                    <h3>预测置信度</h3>
                    {% if files.confidence_plot %}
                    <img src="/download/{{ result_id }}/confidence_plot" class="img-fluid" alt="置信度图">
                    <p class="mt-2">
                        <a href="/download/{{ result_id }}/confidence_plot" class="btn btn-outline-secondary btn-sm">下载图像</a>
                    </p>
                    {% else %}
                    <p>置信度图不可用</p>
                    {% endif %}
                </div>
            </div>
            
            <div class="col-md-5">
                <div class="result-section">
                    <h3>预测报告</h3>
                    <pre>{{ report }}</pre>
                    <p>
                        <a href="/download/{{ result_id }}/report" class="btn btn-outline-secondary btn-sm">下载报告</a>
                    </p>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
    
    # 创建3D查看器模板
    viewer_3d_html = """<!DOCTYPE html>
<html>
<head>
    <title>3D蛋白质结构查看器</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css">
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <style>
        body {
            padding-top: 20px;
            padding-bottom: 20px;
            background-color: #f8f9fa;
        }
        .header {
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        #container3D {
            width: 100%;
            height: 500px;
            position: relative;
            background-color: black;
        }
        .controls {
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>3D蛋白质结构查看器: {{ result_id }}</h1>
            <p>
                <a href="/result/{{ result_id }}" class="btn btn-outline-secondary btn-sm">返回结果页面</a>
                <a href="/" class="btn btn-outline-secondary btn-sm">返回首页</a>
            </p>
        </div>
        
        <div class="row">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-body">
                        <div id="container3D"></div>
                        
                        <div class="controls">
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="form-group">
                                        <label for="style-select">显示样式:</label>
                                        <select id="style-select" class="form-select">
                                            <option value="cartoon">卡通</option>
                                            <option value="line">线框</option>
                                            <option value="stick">棍状</option>
                                            <option value="sphere">球状</option>
                                            <option value="surface">表面</option>
                                        </select>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="form-group">
                                        <label for="color-select">颜色方案:</label>
                                        <select id="color-select" class="form-select">
                                            <option value="spectrum">彩虹</option>
                                            <option value="chain">链</option>
                                            <option value="secondary">二级结构</option>
                                            <option value="residue">残基</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="mt-3">
                                <button id="btn-spin" class="btn btn-primary">旋转</button>
                                <button id="btn-center" class="btn btn-secondary">居中</button>
                                <button id="btn-download" class="btn btn-info">下载PDB</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        $(document).ready(function() {
            let viewer = $3Dmol.createViewer("container3D");
            let spinning = false;
            
            // 加载PDB数据
            $.get('/get_pdb/{{ result_id }}', function(data) {
                viewer.addModel(data, "pdb");
                viewer.setStyle({}, {cartoon: {color: 'spectrum'}});
                viewer.zoomTo();
                viewer.render();
            });
            
            // 样式选择
            $('#style-select').change(function() {
                let style = $(this).val();
                let color = $('#color-select').val();
                
                viewer.setStyle({}, {});  // 清除所有样式
                
                let styleObj = {};
                styleObj[style] = {};
                
                if (color === 'spectrum') {
                    styleObj[style].color = 'spectrum';
                } else if (color === 'chain') {
                    styleObj[style].colorByChain = true;
                } else if (color === 'secondary') {
                    styleObj[style].colorBySSectopn = true;
                } else if (color === 'residue') {
                    styleObj[style].colorByResidue = true;
                }
                
                viewer.setStyle({}, styleObj);
                viewer.render();
            });
            
            // 颜色选择
            $('#color-select').change(function() {
                let style = $('#style-select').val();
                let color = $(this).val();
                
                viewer.setStyle({}, {});  // 清除所有样式
                
                let styleObj = {};
                styleObj[style] = {};
                
                if (color === 'spectrum') {
                    styleObj[style].color = 'spectrum';
                } else if (color === 'chain') {
                    styleObj[style].colorByChain = true;
                } else if (color === 'secondary') {
                    styleObj[style].colorBySSectopn = true;
                } else if (color === 'residue') {
                    styleObj[style].colorByResidue = true;
                }
                
                viewer.setStyle({}, styleObj);
                viewer.render();
            });
            
            // 旋转按钮
            $('#btn-spin').click(function() {
                if (!spinning) {
                    viewer.spin(true);
                    spinning = true;
                    $(this).text('停止旋转');
                } else {
                    viewer.spin(false);
                    spinning = false;
                    $(this).text('旋转');
                }
            });
            
            // 居中按钮
            $('#btn-center').click(function() {
                viewer.zoomTo();
                viewer.render();
            });
            
            // 下载按钮
            $('#btn-download').click(function() {
                window.location.href = '/download/{{ result_id }}/pdb';
            });
        });
    </script>
</body>
</html>
"""
    
    # 写入模板文件
    with open(os.path.join(templates_dir, 'index.html'), 'w') as f:
        f.write(index_html)
    
    with open(os.path.join(templates_dir, 'result.html'), 'w') as f:
        f.write(result_html)
    
    with open(os.path.join(templates_dir, 'viewer_3d.html'), 'w') as f:
        f.write(viewer_3d_html)
    
    print("HTML模板已创建")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AlphaFold演示")
    parser.add_argument('--port', type=int, default=5000, help='服务器端口')
    parser.add_argument('--host', default='127.0.0.1', help='服务器主机')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    # 创建模板文件
    create_templates()
    
    # 启动Flask应用
    print(f"启动AlphaFold演示服务在 http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main() 