#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import requests
import time
import random
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# 示例蛋白质序列（人类血红蛋白Alpha链）
SAMPLE_SEQUENCES = {
    "Hemoglobin_Alpha": "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR",
    "Insulin": "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN",
    "Lysozyme": "MKALIVLGLVLLSVTVQGKVFERCELARTLKRLGMDGYRGISLANWMCLAKWESGYNTRATNYNAGDRSTDYGIFQINSRYWCNDGKTPGAVNACHLSCSALLQDNIADAVACAKRVVRDPQGIRAWVAWRNRCQNRDVRQYVQGCGV"
}

def download_sequence_from_uniprot(uniprot_id, output_dir):
    """
    从UniProt下载蛋白质序列
    """
    print(f"从UniProt下载序列: {uniprot_id}")
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # 保存FASTA文件
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{uniprot_id}.fasta")
        
        with open(output_path, 'w') as f:
            f.write(response.text)
        
        print(f"序列已保存到: {output_path}")
        return output_path
    
    except Exception as e:
        print(f"下载序列失败: {e}")
        return None

def load_sequence_from_fasta(fasta_path):
    """
    从FASTA文件加载蛋白质序列
    """
    try:
        records = list(SeqIO.parse(fasta_path, "fasta"))
        if records:
            return str(records[0].seq)
        else:
            print(f"FASTA文件中没有找到序列: {fasta_path}")
            return None
    except Exception as e:
        print(f"读取FASTA文件失败: {e}")
        return None

def save_sequence_to_fasta(sequence, name, output_dir):
    """
    将蛋白质序列保存为FASTA文件
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{name}.fasta")
    
    # 创建SeqRecord对象
    record = SeqRecord(
        Seq(sequence),
        id=name,
        name=name,
        description=f"Sequence {name} for AlphaFold prediction"
    )
    
    # 保存到FASTA文件
    SeqIO.write(record, output_path, "fasta")
    print(f"序列已保存到: {output_path}")
    return output_path

def get_sample_sequence(name=None):
    """
    获取示例蛋白质序列
    """
    if name and name in SAMPLE_SEQUENCES:
        return name, SAMPLE_SEQUENCES[name]
    else:
        # 随机选择一个示例序列
        name = random.choice(list(SAMPLE_SEQUENCES.keys()))
        return name, SAMPLE_SEQUENCES[name]

def validate_sequence(sequence):
    """
    验证蛋白质序列的有效性
    """
    valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    
    # 检查是否只包含有效的氨基酸
    for aa in sequence:
        if aa not in valid_amino_acids:
            return False, f"序列包含无效字符: {aa}"
    
    # 检查序列长度
    if len(sequence) < 10:
        return False, "序列太短 (少于10个氨基酸)"
    
    if len(sequence) > 1500:
        return False, "序列太长 (超过1500个氨基酸)"
    
    return True, "序列有效"

def prepare_sequence(sequence_input, output_dir, name=None):
    """
    准备蛋白质序列用于预测
    
    sequence_input可以是:
    - 直接的氨基酸序列字符串
    - UniProt ID
    - FASTA文件路径
    - 示例序列名称
    """
    sequence = None
    
    # 检查是否是UniProt ID
    if len(sequence_input) <= 10 and sequence_input.isalnum():
        fasta_path = download_sequence_from_uniprot(sequence_input, output_dir)
        if fasta_path:
            sequence = load_sequence_from_fasta(fasta_path)
            if not name:
                name = sequence_input
    
    # 检查是否是路径
    elif os.path.exists(sequence_input):
        sequence = load_sequence_from_fasta(sequence_input)
        if not name:
            name = os.path.basename(sequence_input).split('.')[0]
    
    # 检查是否是示例序列名称
    elif sequence_input in SAMPLE_SEQUENCES:
        sequence = SAMPLE_SEQUENCES[sequence_input]
        if not name:
            name = sequence_input
    
    # 否则，视为直接的序列字符串
    else:
        sequence = sequence_input.strip()
        if not name:
            name = f"sequence_{int(time.time())}"
    
    # 验证序列
    if sequence:
        is_valid, message = validate_sequence(sequence)
        if is_valid:
            # 保存为FASTA文件
            fasta_path = save_sequence_to_fasta(sequence, name, output_dir)
            return {
                'name': name,
                'sequence': sequence,
                'length': len(sequence),
                'fasta_path': fasta_path
            }
        else:
            print(f"序列无效: {message}")
            return None
    else:
        print("无法解析序列输入")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="准备蛋白质序列用于AlphaFold预测")
    parser.add_argument('--input', help='UniProt ID、序列字符串、FASTA文件路径或示例序列名称')
    parser.add_argument('--name', help='序列名称')
    parser.add_argument('--output', default='../data', help='输出目录')
    parser.add_argument('--sample', action='store_true', help='使用示例序列')
    
    args = parser.parse_args()
    
    if args.sample:
        name, sequence = get_sample_sequence(args.name)
        print(f"使用示例序列: {name}")
        result = prepare_sequence(sequence, args.output, name)
    elif args.input:
        result = prepare_sequence(args.input, args.output, args.name)
    else:
        parser.print_help()
        exit(1)
    
    if result:
        print(f"序列准备完成:")
        print(f"- 名称: {result['name']}")
        print(f"- 长度: {result['length']}")
        print(f"- FASTA文件: {result['fasta_path']}")
        print(f"- 序列: {result['sequence'][:50]}{'...' if len(result['sequence']) > 50 else ''}") 