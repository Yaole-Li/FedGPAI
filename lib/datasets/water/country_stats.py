#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
统计 waterPollution.csv 文件中 Country 列的类型及数量
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
from collections import Counter

def analyze_country_distribution(csv_path):
    """
    分析并统计 CSV 文件中 Country 列的分布情况
    
    Args:
        csv_path: CSV 文件路径
        
    Returns:
        country_counts: 包含每个国家出现次数的字典
        total_rows: 数据总行数
    """
    # 读取 CSV 文件
    print(f"读取 CSV 文件: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 获取数据总行数
    total_rows = len(df)
    print(f"数据总行数: {total_rows}")
    
    # 统计 Country 列中每个值出现的次数
    country_counts = Counter(df['Country'])
    print(f"不同国家数量: {len(country_counts)}")
    
    # 按出现次数降序排序
    sorted_countries = sorted(country_counts.items(), key=lambda x: x[1], reverse=True)
    
    return country_counts, sorted_countries, total_rows

def print_country_stats(sorted_countries, total_rows):
    """打印国家统计信息"""
    print("\n国家分布统计:")
    print("-" * 50)
    print(f"{'国家':<20} {'数量':<10} {'百分比':<10}")
    print("-" * 50)
    
    for country, count in sorted_countries:
        percentage = (count / total_rows) * 100
        print(f"{country:<20} {count:<10} {percentage:.2f}%")

def plot_country_distribution(country_counts, output_path):
    """绘制国家分布的饼图"""
    # 提取前10个国家，其余归为"其他"类别
    top_countries = dict(sorted(country_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    other_count = sum(count for country, count in country_counts.items() if country not in top_countries)
    
    if other_count > 0:
        top_countries['其他'] = other_count
    
    # 创建饼图
    plt.figure(figsize=(10, 8))
    plt.pie(top_countries.values(), labels=top_countries.keys(), autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('水污染数据国家分布')
    
    # 保存图表
    plt.savefig(output_path)
    print(f"国家分布图已保存至: {output_path}")

def save_country_stats(sorted_countries, total_rows, output_path):
    """保存国家统计信息到 CSV 文件"""
    stats_df = pd.DataFrame(sorted_countries, columns=['Country', 'Count'])
    stats_df['Percentage'] = (stats_df['Count'] / total_rows) * 100
    
    stats_df.to_csv(output_path, index=False)
    print(f"国家统计数据已保存至: {output_path}")

def main():
    # 当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # CSV 文件路径
    csv_path = os.path.join(current_dir, 'waterPollution.csv')
    
    # 输出文件路径
    stats_output_path = os.path.join(current_dir, 'country_stats.csv')
    plot_output_path = os.path.join(current_dir, 'country_distribution.png')
    
    # 分析国家分布
    country_counts, sorted_countries, total_rows = analyze_country_distribution(csv_path)
    
    # 打印统计信息
    print_country_stats(sorted_countries, total_rows)
    
    # 保存统计信息
    save_country_stats(sorted_countries, total_rows, stats_output_path)
    
    # 绘制并保存饼图
    plot_country_distribution(country_counts, plot_output_path)

if __name__ == "__main__":
    main()
