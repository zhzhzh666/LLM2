import json
import random
import os
from datetime import datetime, timedelta

# 创建存储目录
output_dir = "inspection_logs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"已创建目录: {output_dir}")

# 巡检点列表
points = [
    "巡检点1_进水pH",
    "巡检点2_沉淀池", 
    "巡检点3_加药系统",
    "巡检点4_澄清池",
    "巡检点5_过滤床",
    "巡检点6_消毒池",
    "巡检点7_清水池",
    "巡检点8_送水泵"
]

def generate_inspection_log(base_date, log_num):
    # 复制基础结构
    visited = points.copy()
    
    # 任务执行状态（大部分成功，偶尔失败）
    tasks = {}
    for point in points:
        if point == "巡检点8_送水泵":
            tasks[point] = "failed"  # 固定故障
        elif point == "巡检点1_进水pH":
            tasks[point] = "success"  # 固定成功但有警告
        else:
            # 其他点95%概率成功，5%概率失败
            tasks[point] = "success" if random.random() > 0.05 else "failed"
    
    # 异常记录
    anomalies = [
        {
            "point": "巡检点8_送水泵",
            "status": "failed",
            "description": "送水泵故障，无法正常运行"
        },
        {
            "point": "巡检点1_进水pH", 
            "status": "warning",
            "description": "存在低风险数据警告，需持续观察pH变化趋势"
        }
    ]
    
    # 随机添加其他异常（5%概率）
    for point in points[1:7]:  # 点2-7
        if random.random() < 0.05:
            anomalies.append({
                "point": point,
                "status": "warning",
                "description": f"{point.split('_')[1]}参数轻微异常，建议关注"
            })
    
    return {
        "log_id": f"LOG_{base_date.strftime('%Y%m%d')}_{log_num:03d}",
        "timestamp": (base_date + timedelta(minutes=log_num*15)).isoformat(),
        "visited_points": visited,
        "tasks_executed": tasks,
        "anomalies": anomalies
    }

# 生成并保存100个独立日志
base_date = datetime(2026, 2, 14, 8, 0)

for i in range(1, 101):  # 从1到100
    log_data = generate_inspection_log(base_date, i)
    
    # 构建文件名，例如：inspection_logs/LOG_20260214_001.json
    filename = f"{log_data['log_id']}.json"
    filepath = os.path.join(output_dir, filename)
    
    # 写入单个文件
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    
    # 可选：打印进度
    if i % 10 == 0:
        print(f"已生成 {i}/100 个文件...")

print(f"\n完成！所有100个日志已单独保存在 '{output_dir}' 文件夹中。")
