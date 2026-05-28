(() => {
  const {createElement: h, useEffect, useRef, useState} = React;
  const root = ReactDOM.createRoot(document.getElementById("root"));
  dayjs.extend(dayjs_plugin_utc);
  const WINDOW_LIMIT = 10;
  const OUTCOME_COLORS = {
    ignored: "rgba(31, 26, 20, 0.38)",
    calibrated: "#2f7f68",
    detected: "#2f5f9f",
    recorded: "#9f4f2a",
  };

  const defaultStatus = {
    selected_device_index: null,
    selected_device_name: "",
    is_running: false,
    is_switching_device: false,
    last_peak_rms: 0,
    last_peak_dbfs: -90,
    detection_window_seconds: 0,
    capture_seconds: 0,
    noise_floor_rms: 0,
    noise_floor_std_rms: 0,
    trigger_threshold_rms: 0,
    noise_floor_dbfs: -90,
    trigger_threshold_dbfs: -90,
    gate_ready: false,
    is_calibrating: false,
    last_gate_open: false,
    recent_detection_windows: [],
    last_error: "",
  };

  function buildQuery(filters) {
    const params = new URLSearchParams();
    ["classification", "start_at", "end_at"].forEach((key) => {
      if (filters[key]) {
        params.set(key, filters[key]);
      }
    });
    return params.toString();
  }

  function buildEventQuery(filters, pagination) {
    const params = new URLSearchParams(buildQuery(filters));
    params.set("page", String(pagination.page || 1));
    params.set("page_size", String(pagination.pageSize || 20));
    return params.toString();
  }

  async function fetchJson(url, options = {}) {
    const response = await fetch(url, {
      ...options,
      headers: {"Accept": "application/json", ...(options.headers || {})},
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "请求失败");
    }
    return payload;
  }

  function formatTimestamp(value) {
    const parsed = dayjs(value);
    return parsed.isValid() ? parsed.local().format("YYYY-MM-DD HH:mm:ss") : value;
  }

  function windowOutcomeLabel(item) {
    const outcome = item.outcome || (item.gate_open ? "detected" : "ignored");
    if (outcome === "recorded") {
      return "已记录";
    }
    if (outcome === "calibrated") {
      return "已校准";
    }
    if (outcome === "detected") {
      return "已检测";
    }
    return "已忽略";
  }

  function windowOutcome(item) {
    return item.outcome || (item.gate_open ? "detected" : "ignored");
  }

  function windowSignature(item) {
    return JSON.stringify({
      timestamp: item.timestamp,
      points: item.points || [],
      rms: item.rms,
      dbfs: item.dbfs,
      trigger_threshold_rms: item.trigger_threshold_rms,
      noise_floor_rms: item.noise_floor_rms,
    });
  }

  function timestampToMs(value) {
    const parsed = Date.parse(value || "");
    return Number.isFinite(parsed) ? parsed : Date.now();
  }

  function StatusCard({label, value}) {
    return h("div", {className: "status-card"}, [
      h("strong", {key: "label"}, label),
      h("div", {key: "value"}, value || "-"),
    ]);
  }

  function StatusCards({status}) {
    return h("section", {className: "status-grid"}, [
      h(StatusCard, {
        key: "running",
        label: "采集状态",
        value: status.is_switching_device ? "切换设备中" : (status.is_running ? "运行中" : "已停止"),
      }),
      h(StatusCard, {
        key: "window",
        label: "检测窗口 / 录制时长",
        value: `${status.detection_window_seconds}s / ${status.capture_seconds}s`,
      }),
      h(StatusCard, {key: "rms", label: "最近峰值 RMS", value: status.last_peak_rms}),
      h(StatusCard, {key: "threshold", label: "分类阈值", value: `${status.trigger_threshold_rms} RMS / ${status.trigger_threshold_dbfs} dBFS`}),
      h(StatusCard, {key: "floor", label: "底噪均值", value: `${status.noise_floor_rms} RMS / ${status.noise_floor_dbfs} dBFS`}),
      h(StatusCard, {key: "std", label: "底噪标准差", value: status.noise_floor_std_rms}),
      h(StatusCard, {
        key: "gate",
        label: "最近窗口",
        value: `${status.last_peak_dbfs} dBFS / ${windowOutcomeLabel((status.recent_detection_windows || []).slice(-1)[0] || {gate_open: status.last_gate_open})}`,
      }),
    ]);
  }

  function AudioDevicePanel({
    status,
    devices,
    canSwitch,
    configuredDeviceName,
    onSwitchDevice,
  }) {
    const [selectedIndex, setSelectedIndex] = useState(
      status.selected_device_index !== null ? String(status.selected_device_index) : "",
    );

    useEffect(() => {
      setSelectedIndex(status.selected_device_index !== null ? String(status.selected_device_index) : "");
    }, [status.selected_device_index]);

    return h("section", {className: "controls"}, [
      h("div", {key: "head", className: "device-panel-head"}, [
        h("div", {key: "copy"}, [
          h("strong", {key: "title", className: "device-panel-title"}, "输入设备"),
          h("div", {key: "hint", className: "muted compact-note"}, "切换会中断当前采集并触发重新校准。"),
        ]),
      ]),
      h("div", {key: "fields", className: "control-fields"}, [
        h("div", {key: "configured"}, [
          h("label", {key: "label"}, "启动默认配置"),
          h("div", {key: "value", className: "device-summary"}, configuredDeviceName || "未配置，默认使用首个可用设备"),
        ]),
        h("div", {key: "select"}, [
          h("label", {key: "label", htmlFor: "audio-device-select"}, "切换到"),
          h("select", {
            key: "input",
            id: "audio-device-select",
            value: selectedIndex,
            disabled: !canSwitch || status.is_switching_device || !devices.length,
            onChange: (event) => setSelectedIndex(event.target.value),
          }, devices.map((device) => h(
            "option",
            {key: String(device.index), value: String(device.index)},
            `#${device.index} ${device.name}`,
          ))),
        ]),
      ]),
      h("div", {key: "actions", className: "actions"}, [
        h("div", {key: "current-device", className: "muted current-device-note"}, `当前：${status.selected_device_name || "未选择"}`),
        h("button", {
          key: "switch",
          className: "button secondary",
          type: "button",
          disabled: (
            !canSwitch
            || status.is_switching_device
            || !selectedIndex
            || String(status.selected_device_index) === selectedIndex
          ),
          onClick: () => onSwitchDevice(Number(selectedIndex)),
        }, status.is_switching_device ? "切换中..." : "切换设备"),
      ]),
    ]);
  }

  function buildTimelineSeries(windows) {
    const waveform = [];
    const markers = [];
    const thresholds = [];
    const floors = [];
    windows.forEach((item) => {
      const startMs = timestampToMs(item.timestamp);
      const durationMs = Math.max(1, Number(item.window_duration_seconds || 1) * 1000);
      const endMs = startMs + durationMs;
      const points = item.points || [];
      points.forEach((point, pointIndex) => {
        const ratio = points.length <= 1 ? 0.5 : pointIndex / (points.length - 1);
        waveform.push({
          id: `${item.timestamp}:wave:${pointIndex}`,
          value: [
            startMs + ratio * durationMs,
            Number(point) || 0,
          ],
        });
      });
      const threshold = Math.max(0, Math.min(1, Number(item.trigger_threshold_rms || 0) / 32768));
      const floor = Math.max(0, Math.min(1, Number(item.noise_floor_rms || 0) / 32768));
      thresholds.push(
        {id: `${item.timestamp}:threshold:p0`, value: [startMs, threshold]},
        {id: `${item.timestamp}:threshold:p1`, value: [endMs, threshold]},
        {id: `${item.timestamp}:threshold:gap0`, value: [endMs, null]},
        {id: `${item.timestamp}:threshold:n0`, value: [startMs, -threshold]},
        {id: `${item.timestamp}:threshold:n1`, value: [endMs, -threshold]},
        {id: `${item.timestamp}:threshold:gap1`, value: [endMs, null]},
      );
      floors.push(
        {id: `${item.timestamp}:floor:p0`, value: [startMs, floor]},
        {id: `${item.timestamp}:floor:p1`, value: [endMs, floor]},
        {id: `${item.timestamp}:floor:gap0`, value: [endMs, null]},
        {id: `${item.timestamp}:floor:n0`, value: [startMs, -floor]},
        {id: `${item.timestamp}:floor:n1`, value: [endMs, -floor]},
        {id: `${item.timestamp}:floor:gap1`, value: [endMs, null]},
      );
      const outcome = windowOutcome(item);
      markers.push({
        id: `${item.timestamp}:marker`,
        value: [startMs + durationMs / 2, -0.5],
        itemStyle: {color: OUTCOME_COLORS[outcome] || OUTCOME_COLORS.ignored},
        name: windowOutcomeLabel(item),
        window: item,
      });
    });
    return {waveform, markers, thresholds, floors};
  }

  function EChartTimeline({windows}) {
    const chartRef = useRef(null);
    const instanceRef = useRef(null);
    const lastSignatureRef = useRef("");
    const windowsRef = useRef([]);

    useEffect(() => {
      if (!chartRef.current || !window.echarts) {
        return undefined;
      }
      instanceRef.current = window.echarts.init(chartRef.current, null, {renderer: "canvas"});
      const handleResize = () => instanceRef.current && instanceRef.current.resize();
      window.addEventListener("resize", handleResize);
      return () => {
        window.removeEventListener("resize", handleResize);
        instanceRef.current.dispose();
        instanceRef.current = null;
      };
    }, []);

    useEffect(() => {
      const chart = instanceRef.current;
      if (!chart) {
        return;
      }
      lastSignatureRef.current = windows.length ? windowSignature(windows[windows.length - 1]) : "";
      windowsRef.current = windows.slice(-WINDOW_LIMIT);
      const {waveform, markers, thresholds, floors} = buildTimelineSeries(windowsRef.current);
      chart.setOption({
        animation: true,
        animationDuration: 260,
        animationDurationUpdate: 420,
        animationEasing: "cubicOut",
        animationEasingUpdate: "cubicOut",
        backgroundColor: "transparent",
        grid: {
          left: 42,
          right: 18,
          top: 22,
          bottom: 52,
        },
        tooltip: {
          trigger: "item",
          formatter: (params) => {
            if (!params.data || !params.data.window) {
              return "";
            }
            const item = params.data.window;
            return [
              windowOutcomeLabel(item),
              `RMS: ${item.rms || 0}`,
              `dBFS: ${item.dbfs || 0}`,
              `阈值: ${item.trigger_threshold_rms || 0} RMS`,
              `底噪: ${item.noise_floor_rms || 0} RMS`,
            ].join("<br>");
          },
        },
        xAxis: {
          type: "time",
          axisLabel: {
            formatter: (value) => dayjs(value).format("HH:mm:ss"),
            color: "rgba(31, 26, 20, 0.58)",
          },
          axisLine: {lineStyle: {color: "rgba(31, 26, 20, 0.22)"}},
          axisTick: {show: false},
          splitLine: {lineStyle: {color: "rgba(31, 26, 20, 0.1)"}},
        },
        yAxis: [
          {
            type: "value",
            scale: true,
            animation: false,
            axisLabel: {
              formatter: (value) => (Math.abs(value) <= 1 ? value.toFixed(2) : ""),
              color: "rgba(31, 26, 20, 0.58)",
            },
            axisLine: {show: false},
            axisTick: {show: false},
            splitLine: {lineStyle: {color: "rgba(31, 26, 20, 0.08)"}},
          },
          {
            type: "value",
            min: -0.6,
            max: 0.5,
            show: false,
          },
        ],
        series: [
          {
            id: "threshold",
            name: "阈值",
            type: "line",
            yAxisIndex: 0,
            data: thresholds,
            symbol: "none",
            connectNulls: false,
            lineStyle: {
              color: "rgba(159, 79, 42, 0.7)",
              width: 1,
              type: "dashed",
            },
            silent: true,
          },
          {
            id: "floor",
            name: "底噪",
            type: "line",
            yAxisIndex: 0,
            data: floors,
            symbol: "none",
            connectNulls: false,
            lineStyle: {
              color: "rgba(31, 26, 20, 0.26)",
              width: 1,
              type: "dotted",
            },
            silent: true,
          },
          {
            id: "waveform",
            name: "波形",
            type: "line",
            yAxisIndex: 0,
            data: waveform,
            symbol: "none",
            showSymbol: false,
            sampling: "lttb",
            lineStyle: {
              color: "rgba(31, 26, 20, 0.78)",
              width: 1,
            },
            emphasis: {disabled: true},
          },
          {
            id: "markers",
            name: "判定",
            type: "scatter",
            yAxisIndex: 1,
            data: markers,
            symbolSize: 9,
            encode: {x: 0, y: 1},
            z: 5,
          },
        ],
      }, {
        notMerge: true,
        lazyUpdate: false,
      });
    }, []);

    useEffect(() => {
      const chart = instanceRef.current;
      if (!chart) {
        return;
      }
      const latest = windows[windows.length - 1];
      const latestSignature = latest ? windowSignature(latest) : "";
      if (!latest || latestSignature === lastSignatureRef.current) {
        return;
      }

      windowsRef.current = [...windowsRef.current, latest].slice(-WINDOW_LIMIT);
      const {waveform, markers, thresholds, floors} = buildTimelineSeries(windowsRef.current);
      chart.setOption({
        series: [
          {id: "threshold", data: thresholds},
          {id: "floor", data: floors},
          {id: "waveform", data: waveform},
          {id: "markers", data: markers},
        ],
      }, {
        notMerge: false,
        lazyUpdate: false,
      });
      lastSignatureRef.current = latestSignature;
    }, [windows]);

    if (!window.echarts) {
      return h("div", {className: "empty"}, "图表库加载中...");
    }
    return h("div", {className: "echart-timeline", ref: chartRef});
  }

  function WaveformMonitor({status, autoRefresh, onAutoRefreshChange, onCalibrate}) {
    const windows = status.recent_detection_windows || [];
    const latestWindow = windows[windows.length - 1] || {};

    return h("section", {className: "monitor-panel"}, [
      h("div", {key: "header", className: "monitor-header"}, [
        h("div", {key: "title"}, [
          h("h2", {key: "h2", className: "monitor-title"}, "最近 10 个判定窗口"),
          h("div", {key: "hint", className: "muted"}, "仅当窗口命中目标标签时，才录制完整片段并写入事件记录"),
        ]),
        h("div", {key: "actions", className: "monitor-actions"}, [
          h("button", {
            key: "calibrate",
            className: "button secondary",
            type: "button",
            disabled: status.is_calibrating,
            onClick: onCalibrate,
          }, status.is_calibrating ? "校准中..." : "校准阈值"),
          h("label", {key: "toggle", className: "refresh-toggle"}, [
            h("input", {
              key: "input",
              type: "checkbox",
              checked: autoRefresh,
              onChange: (event) => onAutoRefreshChange(event.target.checked),
            }),
            "自动刷新",
          ]),
        ]),
      ]),
      windows.length
        ? h("div", {key: "timeline", className: "combined-waveform"}, [
          h(EChartTimeline, {key: "chart", windows}),
          h("div", {key: "legend", className: "timeline-legend"}, [
            h("span", {key: "ignored", className: "legend-item outcome-ignored"}, "已忽略"),
            h("span", {key: "calibrated", className: "legend-item outcome-calibrated"}, "已校准"),
            h("span", {key: "detected", className: "legend-item outcome-detected"}, "已检测"),
            h("span", {key: "recorded", className: "legend-item outcome-recorded"}, "已记录"),
            h("span", {key: "latest", className: "timeline-latest"}, `最新：${windowOutcomeLabel(latestWindow)} / ${latestWindow.rms || 0} RMS`),
          ]),
        ])
        : h("div", {key: "empty", className: "empty"}, "尚无判定窗口数据。"),
    ]);
  }

  function EventFilters({filters, classifications, onSubmit, onReset}) {
    const [draft, setDraft] = useState(filters);

    useEffect(() => {
      setDraft(filters);
    }, [filters]);

    function updateField(name, value) {
      setDraft((current) => ({...current, [name]: value}));
    }

    function submit(event) {
      event.preventDefault();
      onSubmit(draft);
    }

    return h("form", {className: "controls", onSubmit: submit}, [
      h("div", {key: "fields", className: "control-fields"}, [
        h("div", {key: "classification"}, [
          h("label", {key: "label", htmlFor: "classification"}, "分类标签"),
          h("select", {
            key: "select",
            id: "classification",
            value: draft.classification,
            onChange: (event) => updateField("classification", event.target.value),
          }, [
            h("option", {key: "", value: ""}, "全部分类"),
            ...classifications.map((item) => h("option", {key: item, value: item}, item)),
          ]),
        ]),
        h("div", {key: "start"}, [
          h("label", {key: "label", htmlFor: "start_at"}, "开始日期（YYYYMMDD）"),
          h("input", {
            key: "input",
            id: "start_at",
            value: draft.start_at,
            placeholder: "20260512",
            onChange: (event) => updateField("start_at", event.target.value),
          }),
        ]),
        h("div", {key: "end"}, [
          h("label", {key: "label", htmlFor: "end_at"}, "结束日期（YYYYMMDD）"),
          h("input", {
            key: "input",
            id: "end_at",
            value: draft.end_at,
            placeholder: "20260512",
            onChange: (event) => updateField("end_at", event.target.value),
          }),
        ]),
      ]),
      h("div", {key: "actions", className: "actions"}, [
        h("button", {key: "submit", className: "button", type: "submit"}, "应用筛选"),
        h("button", {key: "reset", className: "button secondary", type: "button", onClick: onReset}, "重置"),
        h("a", {key: "export", className: "button secondary", href: `/export?${buildQuery(filters)}`}, "导出 ZIP"),
      ]),
    ]);
  }

  function PaginationControls({pagination, onPageChange}) {
    const page = pagination.page || 1;
    const totalPages = pagination.total_pages || 1;
    const total = pagination.total || 0;
    const pageSize = pagination.page_size || 20;
    const start = total === 0 ? 0 : (page - 1) * pageSize + 1;
    const end = total === 0 ? 0 : Math.min(total, page * pageSize);

    return h("div", {className: "pagination-bar"}, [
      h("div", {key: "summary", className: "muted"}, `第 ${page} / ${totalPages} 页，显示 ${start}-${end} 条，共 ${total} 条`),
      h("div", {key: "actions", className: "pagination-actions"}, [
        h("button", {
          key: "prev",
          className: "button secondary",
          type: "button",
          disabled: page <= 1,
          onClick: () => onPageChange(page - 1),
        }, "上一页"),
        h("button", {
          key: "next",
          className: "button secondary",
          type: "button",
          disabled: page >= totalPages,
          onClick: () => onPageChange(page + 1),
        }, "下一页"),
      ]),
    ]);
  }

  function EventTable({events, pagination, onPageChange}) {
    if (!events.length) {
      return h(React.Fragment, null, [
        h(PaginationControls, {
          key: "pagination-top",
          pagination,
          onPageChange,
        }),
        h("div", {key: "empty", className: "empty"}, "暂无符合条件的事件。"),
      ]);
    }

    return h(React.Fragment, null, [
      h(PaginationControls, {
        key: "pagination-top",
        pagination,
        onPageChange,
      }),
      h("table", {key: "table"}, [
        h("thead", {key: "head"}, h("tr", null, [
          "时间", "峰值 RMS", "时长", "分类", "分数", "候选标签", "试听",
        ].map((item) => h("th", {key: item}, item)))),
        h("tbody", {key: "body"}, events.map((event) => h("tr", {key: `${event.timestamp}-${event.filename}`}, [
          h("td", {key: "timestamp"}, formatTimestamp(event.timestamp)),
          h("td", {key: "rms"}, event.peak_rms),
          h("td", {key: "duration"}, `${event.duration_seconds}s`),
          h("td", {key: "classification"}, h("span", {className: "tag"}, event.classification)),
          h("td", {key: "score"}, event.classification_score),
          h("td", {key: "classes"}, (event.top_classes || []).join(", ")),
          h("td", {key: "audio"}, h("audio", {controls: true, preload: "none", src: event.record_url})),
        ]))),
      ]),
      h(PaginationControls, {
        key: "pagination-bottom",
        pagination,
        onPageChange,
      }),
    ]);
  }

  function Tabs({activeTab, onChange}) {
    return h("nav", {className: "tabs", "aria-label": "页面分区"}, [
      h("button", {
        key: "status",
        className: `tab-button ${activeTab === "status" ? "active" : ""}`,
        type: "button",
        onClick: () => onChange("status"),
      }, "状态"),
      h("button", {
        key: "events",
        className: `tab-button ${activeTab === "events" ? "active" : ""}`,
        type: "button",
        onClick: () => onChange("events"),
      }, "事件列表"),
    ]);
  }

  function App() {
    const [status, setStatus] = useState(defaultStatus);
    const [events, setEvents] = useState([]);
    const [audioDevices, setAudioDevices] = useState([]);
    const [audioDeviceMeta, setAudioDeviceMeta] = useState({
      can_switch: false,
      configured_device_name: "",
      selected_device_index: null,
    });
    const [eventPagination, setEventPagination] = useState({
      page: 1,
      page_size: 20,
      total: 0,
      total_pages: 1,
    });
    const [classifications, setClassifications] = useState([]);
    const [filters, setFilters] = useState({classification: "", start_at: "", end_at: ""});
    const [error, setError] = useState("");
    const [activeTab, setActiveTab] = useState("status");
    const [autoRefresh, setAutoRefresh] = useState(
      () => localStorage.getItem("audio-detect:auto-refresh") !== "0",
    );

    async function loadStatus() {
      setStatus(await fetchJson("/api/status"));
    }

    async function loadEvents(nextFilters = filters, nextPage = eventPagination.page || 1) {
      const query = buildEventQuery(nextFilters, {
        page: nextPage,
        pageSize: eventPagination.page_size || 20,
      });
      const payload = await fetchJson(`/api/events${query ? `?${query}` : ""}`);
      setEvents(payload.events || []);
      setEventPagination(payload.pagination || {
        page: nextPage,
        page_size: eventPagination.page_size || 20,
        total: 0,
        total_pages: 1,
      });
    }

    async function loadClassifications() {
      const payload = await fetchJson("/api/classifications");
      setClassifications(payload.classifications || []);
    }

    async function loadAudioDevices() {
      const payload = await fetchJson("/api/audio-devices");
      setAudioDevices(payload.devices || []);
      setAudioDeviceMeta({
        can_switch: Boolean(payload.can_switch),
        configured_device_name: payload.configured_device_name || "",
        selected_device_index: payload.selected_device_index,
      });
    }

    async function refresh(nextFilters = filters, nextPage = eventPagination.page || 1) {
      try {
        setError("");
        await Promise.all([loadStatus(), loadEvents(nextFilters, nextPage)]);
      } catch (err) {
        setError(err.message);
      }
    }

    useEffect(() => {
      refresh();
      Promise.all([loadClassifications(), loadAudioDevices()]).catch((err) => setError(err.message));
    }, []);

    useEffect(() => {
      localStorage.setItem("audio-detect:auto-refresh", autoRefresh ? "1" : "0");
      if (!autoRefresh || !status.detection_window_seconds) {
        return undefined;
      }
      const timer = window.setInterval(() => {
        refresh(filters, eventPagination.page || 1);
      }, Number(status.detection_window_seconds) * 1000);
      return () => window.clearInterval(timer);
    }, [autoRefresh, status.detection_window_seconds, filters, eventPagination.page]);

    function applyFilters(nextFilters) {
      setFilters(nextFilters);
      loadEvents(nextFilters, 1).then(() => setError("")).catch((err) => setError(err.message));
    }

    function resetFilters() {
      const emptyFilters = {classification: "", start_at: "", end_at: ""};
      setFilters(emptyFilters);
      loadEvents(emptyFilters, 1).then(() => setError("")).catch((err) => setError(err.message));
    }

    function changeEventPage(nextPage) {
      loadEvents(filters, nextPage).then(() => setError("")).catch((err) => setError(err.message));
    }

    async function calibrateThreshold() {
      try {
        setError("");
        const nextStatus = await fetchJson("/api/calibrate", {method: "POST"});
        setStatus(nextStatus);
      } catch (err) {
        setError(err.message);
      }
    }

    async function switchAudioDevice(deviceIndex) {
      try {
        setError("");
        setStatus((current) => ({...current, is_switching_device: true}));
        const nextStatus = await fetchJson("/api/audio-device", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({device_index: deviceIndex}),
        });
        setStatus(nextStatus);
        await loadAudioDevices();
      } catch (err) {
        setError(err.message);
        await Promise.all([loadStatus(), loadAudioDevices()]);
      }
    }

    return h(React.Fragment, null, [
      h("h1", {key: "title"}, "噪音监测面板"),
      h("p", {key: "intro"}, "系统会按固定检测窗口采集音频并运行分类"),
      h(Tabs, {key: "tabs", activeTab, onChange: setActiveTab}),
      error ? h("div", {key: "error", className: "error"}, error) : null,
      activeTab === "status"
        ? h("section", {key: "status-tab", className: "tab-panel"}, [
          h(StatusCards, {key: "status", status}),
          h(AudioDevicePanel, {
            key: "device-panel",
            status,
            devices: audioDevices,
            canSwitch: audioDeviceMeta.can_switch,
            configuredDeviceName: audioDeviceMeta.configured_device_name,
            onSwitchDevice: switchAudioDevice,
          }),
          status.last_error ? h("p", {key: "warning"}, `音频后端警告：${status.last_error}`) : null,
          h(WaveformMonitor, {
            key: "monitor",
            status,
            autoRefresh,
            onAutoRefreshChange: setAutoRefresh,
            onCalibrate: calibrateThreshold,
          }),
        ])
        : h("section", {key: "events-tab", className: "tab-panel"}, [
          h(EventFilters, {
            key: "filters",
            filters,
            classifications,
            onSubmit: applyFilters,
            onReset: resetFilters,
          }),
          h("p", {key: "muted", className: "muted"}, "当前事件列表只包含已保留的录音"),
          h(EventTable, {
            key: "events",
            events,
            pagination: eventPagination,
            onPageChange: changeEventPage,
          }),
        ]),
    ]);
  }

  root.render(h(App));
})();
