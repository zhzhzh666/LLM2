#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能水处理厂巡检计划生成器 (批量版 - 修复优先级逻辑)
- 修复：强制将失败任务点置顶，解决 SRTP 合格率低的问题
- 功能：读取 inspection_logs 文件夹中的独立日志，生成独立的计划和路线图
"""
import os
import json
import math
import re
import dashscope
from dashscope import Generation

# 绘图库导入
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️ 未检测到 matplotlib，将跳过地图可视化。请运行: pip install matplotlib")

# ======================
# 配置与常量
# ======================
INPUT_DIR = "inspection_logs"          # 输入：历史日志文件夹
OUTPUT_DIR = "inspection_plans"        # 输出：生成的计划和图片文件夹
API_KEY_ENV = "DASHSCOPE_API_KEY"

# 初始化 API
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
dashscope.api_key = os.getenv(API_KEY_ENV)

if not dashscope.api_key:
    print(f"❌ 错误：未找到环境变量 {API_KEY_ENV}")
    print("请设置 export DASHSCOPE_API_KEY='your_key'")
    exit(1)

# ======================
# 1. 地图数据定义
# ======================
MAP_DATA = {
    "resolution": 0.5,
    "start": {"name": "入口", "x": 1.0, "y": 1.0, "type": "start"},
    "end": {"name": "出口", "x": 19.0, "y": 1.0, "type": "end"},
    "points": {
        "巡检点1_进水pH": {"x": 2.5, "y": 14.5, "task": 1, "desc": "进水pH计检测"},
        "巡检点2_沉淀池": {"x": 8.5, "y": 13.5, "task": 1, "desc": "沉淀池液位/状态"},
        "巡检点3_加药系统": {"x": 4.5, "y": 6.5, "task": 2, "desc": "加药泵状态检查"},
        "巡检点4_澄清池": {"x": 10.5, "y": 10.5, "task": 1, "desc": "圆形澄清池取样"},
        "巡检点5_过滤床": {"x": 16.5, "y": 14.5, "task": 1, "desc": "过滤床压差监测"},
        "巡检点6_消毒池": {"x": 14.5, "y": 7.5, "task": 1, "desc": "消毒余氯检测"},
        "巡检点7_清水池": {"x": 18.5, "y": 7.5, "task": 2, "desc": "清水池水位检查"},
        "巡检点8_送水泵": {"x": 16.5, "y": 3.5, "task": 2, "desc": "送水泵振动/温度"},
    },
    "obstacles": [
        {"name": "障碍区域B", "x": 2.0, "y": 2.0, "w": 3.0, "h": 2.0},
        {"name": "障碍区域A", "x": 14.0, "y": 4.0, "w": 6.0, "h": 2.5},
        {"name": "障碍区域C", "x": 18.0, "y": 7.0, "w": 2.0, "h": 2.0},
    ]
}

def get_all_patrol_points():
    """构建完整的巡检点字典"""
    points_dict = {}
    s = MAP_DATA["start"]
    points_dict[s["name"]] = {"name": s["name"], "x": s["x"], "y": s["y"], "z": 0.0, "arm_task": 0}
    
    for name, info in MAP_DATA["points"].items():
        points_dict[name] = {
            "name": name, "x": info["x"], "y": info["y"], "z": 0.0, 
            "arm_task": info["task"], "desc": info.get("desc", "")
        }
        
    e = MAP_DATA["end"]
    points_dict[e["name"]] = {"name": e["name"], "x": e["x"], "y": e["y"], "z": 0.0, "arm_task": 0}
    return points_dict

# ======================
# 2. 核心算法 (新增优先级强制逻辑)
# ======================
def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def enforce_priority_logic(middle_waypoints, last_log):
    """
    【关键修复】强制将上次失败的任务点移动到列表最前端
    规则：入口 -> [所有失败点] -> [其他正常点 (贪心排序)] -> 出口
    """
    if not last_log:
        return middle_waypoints
    
    failed_points = set()
    
    # 1. 识别失败点
    tasks_executed = last_log.get("tasks_executed", {})
    for point_name, status in tasks_executed.items():
        if status == "failed":
            failed_points.add(point_name)
    
    if not failed_points:
        # 如果没有失败点，直接返回原列表（后续会进行贪心排序）
        return middle_waypoints
    
    # 2. 分离失败点和正常点
    failed_list = []
    normal_list = []
    
    for wp in middle_waypoints:
        name = wp["point_name"]
        if name in failed_points:
            failed_list.append(wp)
        else:
            normal_list.append(wp)
    
    # 3. 对正常点进行贪心排序 (保持原有的路径优化)
    # 注意：这里我们以“最后一个失败点”或“入口”为起点开始排序正常点
    # 为了简化，我们暂时只对 normal_list 内部进行贪心，或者以入口为起点重新排所有正常点
    # 更优策略：将 failed_list 放在最前，normal_list 接在后面并进行贪心优化
    
    # 如果存在正常点，对它们进行贪心排序，起点视为“最后一个失败点”的位置（如果无失败点则为入口）
    if normal_list:
        # 确定贪心起始坐标
        if failed_list:
            start_pos = failed_list[-1]["position"][:2]
        else:
            # 理论上不会走到这，因为上面判断了 failed_points 存在
            start_pos = [1.0, 1.0] 
            
        ordered_normal = []
        unvisited = normal_list[:]
        current_pos = start_pos
        
        while unvisited:
            nearest = min(unvisited, key=lambda w: euclidean_distance(current_pos, w["position"][:2]))
            ordered_normal.append(nearest)
            current_pos = nearest["position"][:2]
            unvisited.remove(nearest)
        
        # 合并：失败点 (保持原相对顺序或按任务紧急度) + 排序后的正常点
        # 这里简单保持 failed_list 的原始顺序，也可以对 failed_list 内部再做一次贪心
        return failed_list + ordered_normal
    
    return failed_list

def solve_tsp_with_fixed_ends(middle_waypoints, point_dict):
    """贪心算法生成路径：入口 -> 中间点 (已预处理优先级) -> 出口"""
    if "入口" not in point_dict or "出口" not in point_dict:
        raise ValueError("缺少入口或出口定义")
        
    start_node = point_dict["入口"]
    end_node = point_dict["出口"]
    
    start_point = {
        "point_name": "入口",
        "position": [start_node["x"], start_node["y"], start_node["z"]],
        "action": "arrive",
        "arm_task_type": 0
    }
    
    end_point = {
        "point_name": "出口",
        "position": [end_node["x"], end_node["y"], end_node["z"]],
        "action": "arrive",
        "arm_task_type": 0
    }
    
    if not middle_waypoints:
        return [start_point, end_point]
    
    # 注意：middle_waypoints 此时已经过 enforce_priority_logic 处理，失败点已在最前
    # 这里不再进行全局贪心，以免打乱失败的优先级
    # 但如果需要优化失败点内部或正常点内部的顺序，可以在 enforce_priority_logic 中完成
    # 此处直接返回拼接结果，确保顺序不被打乱
    return [start_point] + middle_waypoints + [end_point]

def calculate_path_stats(waypoints):
    """计算路径统计"""
    stats = {"total_distance": 0.0, "segments": [], "total_points": len(waypoints)}
    for i in range(len(waypoints) - 1):
        p1 = waypoints[i]["position"][:2]
        p2 = waypoints[i+1]["position"][:2]
        dist = euclidean_distance(p1, p2)
        stats["total_distance"] += dist
        stats["segments"].append({
            "from": waypoints[i]["point_name"],
            "to": waypoints[i+1]["point_name"],
            "distance_m": round(dist, 2)
        })
    stats["total_distance"] = round(stats["total_distance"], 2)
    return stats

# ======================
# 3. LLM 交互 (强化 Prompt)
# ======================
def build_prompt(point_dict, last_log):
    task_points = {k: v for k, v in point_dict.items() if k not in ["入口", "出口"]}
    
    map_lines = []
    for name, p in task_points.items():
        task_desc = {0: "无", 1: "检测/取样", 2: "设备检查/维护"}.get(p["arm_task"], "未知")
        desc = p.get("desc", "")
        map_lines.append(f"- {name}: ({p['x']:.1f}, {p['y']:.1f}) [{task_desc}] ({desc})")
    
    log_info = "无历史记录"
    failed_points_names = []
    
    if last_log.get("visited_points"):
        log_lines = []
        for pt in last_log["visited_points"]:
            status = last_log["tasks_executed"].get(pt, "未知")
            log_lines.append(f"- {pt}: {status}")
            if status == "failed":
                failed_points_names.append(pt)
        
        log_info = "\n".join(log_lines)
        if last_log.get("anomalies"):
            log_info += "\n⚠️ 异常:\n" + "\n".join(f"  • {a}" for a in last_log["anomalies"])

    # 强化指令：明确要求 LLM 在返回列表时将失败点放在最前面
    priority_instruction = ""
    if failed_points_names:
        priority_instruction = f"\n❗ 重要约束：以下任务点上次执行失败：{', '.join(failed_points_names)}。\n你的 inspection_plan 列表中，这些点**必须**排在所有其他点的最前面（仅次于隐含的入口）！"

    prompt = f"""你是一个智能水处理厂巡检调度专家。
【厂区地图点位】
{chr(10).join(map_lines)}

【上次巡检日志】
{log_info}
{priority_instruction}

【调度规则】
1. **最高优先级**：优先重试上次失败 (failed) 的任务点。在输出的 JSON 列表中，失败点必须位于索引 0 的位置（即紧接在入口之后）。
2. 成功的点本次可跳过，除非是周期性强制检查点。
3. 路径必须从 "入口" 开始，到 "出口" 结束。
4. 输出严格 JSON 格式，不要 Markdown 标记。
   格式示例：
   {{
     "decision_reason": "上次送水泵检查失败，需优先重试。",
     "inspection_plan": [
       {{"point_name": "巡检点8_送水泵", "action": "task_2"}}, 
       {{"point_name": "巡检点6_消毒池", "action": "task_1"}}
     ]
   }}
"""
    return prompt

def call_llm(prompt):
    try:
        response = Generation.call(
            model="qwen3-max",
            messages=[{"role": "user", "content": prompt}],
            result_format="message",
            temperature=0.1
        )
        if response.status_code == 200:
            content = response.output.choices[0].message.content
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        else:
            print(f"API Error: {response.code}")
    except Exception as e:
        print(f"LLM Call Failed: {e}")
    return None

# ======================
# 4. 地图绘制
# ======================
def draw_water_plant_map(waypoints, output_file, title_suffix=""):
    if not HAS_MATPLOTLIB:
        return

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor('#f0f0f0')
    ax.grid(True, which='both', color='#cccccc', linewidth=0.5, linestyle='-')
    ax.set_xticks(range(0, 22, 2))
    ax.set_yticks(range(0, 16, 2))
    ax.set_axisbelow(True)

    for obs in MAP_DATA["obstacles"]:
        rect = patches.Rectangle((obs["x"], obs["y"]), obs["w"], obs["h"],
            linewidth=2, edgecolor='#d9534f', facecolor='#f8d7da', hatch='///')
        ax.add_patch(rect)
        ax.text(obs["x"] + obs["w"]/2, obs["y"] + obs["h"]/2, obs["name"],
                ha='center', va='center', color='#d9534f', fontsize=9, fontweight='bold')

    equipment_configs = [
        ((6, 12), 5, 3, "沉淀池 1&2"), ((14, 13), 5, 2.5, "过滤床"),
        ((13, 6.5), 3, 2, "消毒池"), ((17, 6.5), 3, 2, "清水池"),
        ((14, 2), 6, 1.5, "送水泵组")
    ]
    for (x, y), w, h, label in equipment_configs:
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False, edgecolor='#0275d8', linewidth=2))
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', color='#0275d8', fontweight='bold', fontsize=8 if len(label)>4 else 10)

    if len(waypoints) > 1:
        xs = [p["position"][0] for p in waypoints]
        ys = [p["position"][1] for p in waypoints]
        ax.plot(xs, ys, color='#28a745', linewidth=2.5, linestyle='--', marker='o', markersize=6, label='巡检路线')
        for i in range(len(xs) - 1):
            dx, dy = xs[i+1] - xs[i], ys[i+1] - ys[i]
            if math.hypot(dx, dy) > 0.5:
                ax.annotate('', xy=(xs[i+1], ys[i+1]), xytext=(xs[i] + dx * 0.6, ys[i] + dy * 0.6),
                            arrowprops=dict(arrowstyle='->', color='#28a745', lw=2))

    for wp in waypoints:
        name = wp["point_name"]
        x, y = wp["position"][0], wp["position"][1]
        if name == "入口":
            color, label, zorder = '#28a745', "🚩 入口", 10
        elif name == "出口":
            color, label, zorder = '#dc3545', "🏁 出口", 10
        else:
            task_type = wp.get("arm_task_type", 0)
            color = '#ffc107' if task_type == 2 else '#17a2b8'
            label = f"📍 {name.replace('巡检点', '')}"
            zorder = 5
            
        ax.scatter(x, y, c=color, s=150, zorder=zorder, edgecolors='white', linewidth=2)
        ax.text(x, y + 0.4, label, ha='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=color))

    stats = calculate_path_stats(waypoints)
    info_text = f"📊 巡检统计\n总里程：{stats['total_distance']:.2f} m\n任务点数：{stats['total_points'] - 2}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlim(-1, 21)
    ax.set_ylim(-1, 16)
    ax.set_aspect('equal')
    title = "🏭 智能水处理厂自动巡检路线图"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title, fontsize=16, pad=20)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

# ======================
# 5. 主程序入口 (集成优先级修复)
# ======================
def process_single_log(log_data, file_index, point_dict):
    """处理单个日志并生成计划和地图"""
    log_id = log_data.get("log_id", f"UNKNOWN_{file_index}")
    print(f"   [{file_index}] 处理日志: {log_id} ...")

    # 1. 调用 LLM
    prompt = build_prompt(point_dict, log_data)
    llm_result = call_llm(prompt)
    
    if not llm_result:
        print(f"      ⚠️ LLM 失败，使用默认全量计划。")
        default_plan = [{"point_name": k, "action": "task_1" if v["arm_task"]==1 else "task_2"} 
                        for k, v in point_dict.items() if k not in ["入口", "出口"]]
        llm_result = {"decision_reason": "默认全量巡检 (LLM Failover)", "inspection_plan": default_plan}
    
    # 2. 解析路径数据 (转换为带坐标的 waypoint 对象)
    middle_points_raw = []
    action_map = {"arrive": "arrive", "task_1": "execute_arm_task_1", "task_2": "execute_arm_task_2"}
    
    for item in llm_result.get("inspection_plan", []):
        name = item.get("point_name")
        if name and name in point_dict and name not in ["入口", "出口"]:
            p = point_dict[name]
            middle_points_raw.append({
                "point_name": name,
                "position": [p["x"], p["y"], p["z"]],
                "action": action_map.get(item.get("action"), "arrive"),
                "arm_task_type": p["arm_task"]
            })
            
    # 3. 【关键修复】强制执行优先级逻辑
    # 在传入 TSP 求解器之前，先根据 last_log 调整顺序
    middle_points_ordered = enforce_priority_logic(middle_points_raw, log_data)
    
    # 4. 生成最终路径 (添加入口和出口)
    final_path = solve_tsp_with_fixed_ends(middle_points_ordered, point_dict)
    path_stats = calculate_path_stats(final_path)
    
    # 5. 构建输出数据
    output_data = {
        "source_log_id": log_id,
        "timestamp": log_data.get("timestamp", ""),
        "decision_reason": llm_result.get("decision_reason", ""),
        "route_summary": {
            "total_distance_m": path_stats["total_distance"],
            "total_waypoints": path_stats["total_points"],
            "segments": path_stats["segments"]
        },
        "inspection_plan": final_path
    }
    
    # 6. 保存 JSON
    plan_filename = f"plan_{log_id}.json"
    plan_path = os.path.join(OUTPUT_DIR, plan_filename)
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # 7. 绘制地图
    map_filename = f"route_{log_id}.png"
    map_path = os.path.join(OUTPUT_DIR, map_filename)
    draw_water_plant_map(final_path, map_path, title_suffix=f"基于日志: {log_id}")
    
    return True

def main():
    print("🚀 正在启动批量巡检计划生成器 (已修复优先级逻辑)...")
    
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 错误：未找到输入目录 '{INPUT_DIR}'。请先运行日志生成脚本。")
        return
    
    log_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.json')])
    if not log_files:
        print(f"❌ 错误：'{INPUT_DIR}' 中没有找到 JSON 文件。")
        return
    
    print(f"📂 发现 {len(log_files)} 个历史日志文件。")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 已创建输出目录: {OUTPUT_DIR}")
    
    point_dict = get_all_patrol_points()
    
    success_count = 0
    for i, filename in enumerate(log_files, 1):
        filepath = os.path.join(INPUT_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                log_data = json.load(f)
            
            process_single_log(log_data, i, point_dict)
            success_count += 1
            
            if i % 10 == 0:
                print(f"   ... 已处理 {i}/{len(log_files)} 个 ...")
                
        except Exception as e:
            print(f"   ❌ 处理文件 {filename} 时出错: {e}")
    
    print(f"\n✅ 完成！成功生成 {success_count}/{len(log_files)} 个巡检计划和路线图。")
    print(f"💾 结果保存在: {OUTPUT_DIR}/")
    print("💡 提示：已启用强制优先级逻辑，失败点将自动置顶。")

if __name__ == "__main__":
    main()