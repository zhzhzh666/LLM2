# LLM2
先运行虚拟环境
pip install openai dashscope matplotlib tqdm requests python-dotenv
先运行create100.py生成巡检文件，我这边有但是太多了无法上传
old里面放的是用qwen的api的巡检方案old/qwen里面的文件是最新一次的巡检方案
然后
python generate_inspection_plan.py用来给出新的方案，评估前
 export OPENAI_API_KEY="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"
 or
 export DASHSCOPE_API_KEY="sk-ede645eb2ecb4ede99c8adce9f3b0f5e"
 上面那个是chatgpt的，下面的是qwen的密钥
 然后运行
 python evaluate_plans.py进行评估
 当运行
