#!/bin/bash
# 轴承故障数据分析一键运行脚本
# Bearing Fault Data Analysis - One-Click Run Script

echo "=========================================="
echo "轴承故障数据分析程序"
echo "Bearing Fault Data Analysis Program"
echo "=========================================="

# 检查虚拟环境是否存在
if [ ! -d "venv" ]; then
    echo "创建Python虚拟环境..."
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "激活虚拟环境..."
echo "Activating virtual environment..."
source venv/bin/activate

# 安装依赖
echo "安装依赖包..."
echo "Installing dependencies..."
pip install matplotlib numpy

# 运行分析程序
echo "运行数据分析..."
echo "Running data analysis..."
python create_english_pie_chart.py

echo ""
echo "=========================================="
echo "分析完成！生成的文件："
echo "Analysis completed! Generated files:"
echo "=========================================="
echo "📊 扇形图 | Pie Charts:"
echo "   - 数据类型分布扇形图.png (中文版)"
echo "   - Bearing_Fault_Distribution_Chart.png (英文版)"
echo ""
echo "📄 报告 | Reports:"
echo "   - Bearing_Fault_Analysis_Report.txt (详细英文报告)"
echo "   - 数据分析总结报告.md (中英文总结)"
echo ""
echo "💻 代码 | Code:"
echo "   - create_english_pie_chart.py (主分析程序)"
echo "   - analyze_data.py (简化版)"
echo "   - data_analysis_pie_chart.py (完整版)"
echo ""
echo "🎯 数据统计结果："
echo "📈 Data Statistics:"
echo "   - OR类(外圈故障): 77个文件 (47.8%)"
echo "   - B类(球轴承故障): 40个文件 (24.8%)"
echo "   - IR类(内圈故障): 40个文件 (24.8%)"
echo "   - N类(正常状态): 4个文件 (2.5%)"
echo "   - 总计: 161个文件"
echo "=========================================="