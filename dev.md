# 开发说明

## 技术架构与主要实现

整体架构：

- [app.py](/Users/bytedance/Repo/audio-detect/app.py)：程序入口。
- [audio_detect/main.py](/Users/bytedance/Repo/audio-detect/audio_detect/main.py)：启动配置、日志、存储、分类器、采集器和 Web 服务。
- [audio_detect/collector.py](/Users/bytedance/Repo/audio-detect/audio_detect/collector.py)：音频采集、动态门控、分类触发、录音保留。
- [audio_detect/classifier.py](/Users/bytedance/Repo/audio-detect/audio_detect/classifier.py)：YAMNet TFLite 分类封装。
- [audio_detect/storage.py](/Users/bytedance/Repo/audio-detect/audio_detect/storage.py)：SQLite 事件存储、保留策略、ZIP 导出。
- [audio_detect/web.py](/Users/bytedance/Repo/audio-detect/audio_detect/web.py)：Flask Web 服务和 JSON API。
- [templates/index.html](/Users/bytedance/Repo/audio-detect/templates/index.html)：React CDN 前端入口。
- [static/app.js](/Users/bytedance/Repo/audio-detect/static/app.js)：React UMD 前端应用。
- [static/styles.css](/Users/bytedance/Repo/audio-detect/static/styles.css)：前端样式。
- [config/default.yaml](/Users/bytedance/Repo/audio-detect/config/default.yaml)：默认运行配置。

Web 架构：

- Flask 负责 API、录音文件访问和 ZIP 导出。
- 前端采用 React UMD，从 CDN 加载，无需 Node.js 构建链。
- `/` 返回静态前端入口。
- `/api/status` 返回采集状态、动态阈值和最近判定窗口波形。
- `/api/events` 返回筛选后的事件列表。
- `/api/classifications` 返回事件中已有的分类标签。
- `/records/<filename>` 提供录音播放。
- `/export` 导出当前筛选结果。

检测链路：

1. `AudioInput` 从真实输入设备读取 int16 音频块。
2. `CollectorService` 按 `detection.window_seconds` 聚合检测窗口。
3. `_NoiseFloorGate` 把窗口 RMS 转为 dBFS，并基于长窗口低分位数估计底噪。
4. 未超过动态阈值的窗口标记为 `ignored`。
5. 超过阈值的窗口进入分类，未命中保留标签标记为 `detected`。
6. 命中保留标签后继续采集 `capture_seconds`，将检测窗口和录制窗口合并保存。
7. 保存成功后窗口标记为 `recorded`，事件写入 SQLite。

动态门控设计：

- 内部维护约 3 分钟历史，不暴露为用户配置。
- 使用低分位数估计底噪，避免短时突发影响基线。
- 明显高于底噪的窗口不参与底噪学习，避免持续前景噪音把阈值抬高。
- 使用“相对底噪增量 + 绝对下限”的双条件触发分类。
- 启动初期需要预热窗口，避免冷启动误判。

## 测试

使用项目运行时相同的 Python 解释器安装 pytest：

```bash
python3 -m pip install pytest
```

在 macOS Command Line Tools Python 环境下，pytest 可能会安装到
`/Users/bytedance/Library/Python/3.9/bin`，该目录不一定在 `PATH` 中。
建议通过 Python 模块入口运行测试：

```bash
python3 -m pytest -q
```

当前完整测试结果：

```text
30 passed
```
