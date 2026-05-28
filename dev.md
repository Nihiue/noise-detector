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
- [static/app.js](/Users/bytedance/Repo/audio-detect/static/app.js)：React UMD 前端应用，使用 ECharts 渲染最近判定窗口时间轴。
- [static/styles.css](/Users/bytedance/Repo/audio-detect/static/styles.css)：前端样式。
- [config/default.yaml](/Users/bytedance/Repo/audio-detect/config/default.yaml)：默认运行配置。

Web 架构：

- Flask 负责 API、录音文件访问和 ZIP 导出。
- 前端采用 React UMD 和 ECharts，从 CDN 加载，无需 Node.js 构建链。
- `/` 返回静态前端入口。
- `/api/status` 返回采集状态、当前校准阈值和最近判定窗口波形；每个窗口包含 `timestamp` 和 `window_duration_seconds` 供前端按时间轴绘制。
- `/api/calibrate` 请求采集器使用下一个判断窗口重新校准阈值。
- `/api/events` 返回筛选后的事件列表。
- `/api/classifications` 返回事件中已有的分类标签。
- `/records/<filename>` 提供录音播放。
- `/export` 导出当前筛选结果。

检测链路：

1. `AudioInput` 从真实输入设备读取 int16 音频块。
2. `CollectorService` 按 `detection.window_seconds` 聚合检测窗口。
3. `_NoiseFloorGate` 启动时把第一个判断窗口分桶计算 RMS 均值和标准差，并用 `均值 + K 个标准差` 作为阈值。
4. 未超过当前校准阈值的窗口标记为 `ignored`。
5. 超过阈值的窗口直接进入分类；未命中保留标签标记为 `detected`。
6. 命中保留标签后，触发窗口立即标记为 `recorded`，随后继续采集 `capture_seconds`。
7. 录制期间仍按检测窗口分批发布 `recorded` 判定窗口，不再对这些录制期窗口运行分类。
8. 录制完成后，将触发窗口和录制期窗口的原始 PCM 合并保存为一段录音，并用触发窗口的分类结果写入 SQLite。

阈值门控设计：

- 启动后第一个判断窗口只用于校准底噪，不进入分类。
- 底噪显示为校准窗口内分桶 RMS 的平均值，阈值为平均值加 `detection.threshold_stddev_multiplier` 倍标准差，再加 `detection.threshold_rms_offset` 固定 RMS 偏移量。
- 当前窗口 RMS 只要高于校准阈值，就进入分类。
- Web UI 的“校准阈值”按钮会请求采集器将下一个判断窗口作为新底噪样本并刷新阈值。
- 阈值校准仍在采集线程内完成，避免 Web 请求线程和采集线程同时读取音频设备。

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
35 passed
```
