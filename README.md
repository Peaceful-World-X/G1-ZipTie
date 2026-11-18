# G1 穿扎带仿真

## 目录

```text
G1-ZipTie/
├─ src/                  MuJoCo 仿真源码与工具
├─ tests/                仿真与回放测试
├─ assets/               机器人模型、网格与材质
├─ scenes/               MuJoCo 场景
├─ references/           参考画面与关节数据
├─ recordings/           演示视频、回放轨迹与验证结果
├─ layout.json           工作台布局与初始姿态
└─ camera_mounts.json    腕部相机安装参数
```

## 常用命令（在根目录运行）

```powershell
.\启动仿真.bat           # 空格播放，拖动进度条跳转
.\录制视频.bat           # 导出 MP4
.\播放演示视频.bat       # 播放已导出的视频

.\setup.ps1                                              # 首次安装与建场景（需 uv、Git）
.\.venv\Scripts\python.exe src\task_sequence.py          # 生成动作轨迹
.\.venv\Scripts\python.exe -m unittest discover -s tests  # 测试
```
