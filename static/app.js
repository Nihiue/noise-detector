(() => {
  const {createElement: h, useEffect, useState} = React;
  const root = ReactDOM.createRoot(document.getElementById("root"));
  dayjs.extend(dayjs_plugin_utc);

  const defaultStatus = {
    selected_device_index: null,
    selected_device_name: "",
    is_running: false,
    last_peak_rms: 0,
    last_peak_dbfs: -90,
    detection_window_seconds: 0,
    capture_seconds: 0,
    noise_floor_dbfs: -90,
    trigger_threshold_dbfs: -90,
    gate_ready: false,
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

  async function fetchJson(url) {
    const response = await fetch(url, {headers: {"Accept": "application/json"}});
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

  function dbfsToY(dbfs) {
    const clamped = Math.max(-90, Math.min(0, Number(dbfs)));
    return 72 - ((clamped + 90) / 90) * 72;
  }

  function windowOutcomeLabel(item) {
    const outcome = item.outcome || (item.gate_open ? "detected" : "ignored");
    if (outcome === "recorded") {
      return "已记录";
    }
    if (outcome === "detected") {
      return "已检测";
    }
    return "已忽略";
  }

  function StatusCard({label, value}) {
    return h("div", {className: "status-card"}, [
      h("strong", {key: "label"}, label),
      h("div", {key: "value"}, value || "-"),
    ]);
  }

  function StatusCards({status}) {
    const device = status.selected_device_index !== null
      ? `#${status.selected_device_index} ${status.selected_device_name}`
      : status.selected_device_name;
    return h("section", {className: "status-grid"}, [
      h(StatusCard, {key: "device", label: "输入设备", value: device}),
      h(StatusCard, {key: "running", label: "采集状态", value: status.is_running ? "运行中" : "已停止"}),
      h(StatusCard, {
        key: "window",
        label: "检测窗口 / 录制时长",
        value: `${status.detection_window_seconds}s / ${status.capture_seconds}s`,
      }),
      h(StatusCard, {key: "rms", label: "最近峰值 RMS", value: status.last_peak_rms}),
      h(StatusCard, {key: "threshold", label: "当前门限", value: `${status.trigger_threshold_dbfs} dBFS`}),
      h(StatusCard, {key: "floor", label: "估计底噪", value: `${status.noise_floor_dbfs} dBFS`}),
      h(StatusCard, {
        key: "gate",
        label: "最近窗口",
        value: `${status.last_peak_dbfs} dBFS / ${windowOutcomeLabel((status.recent_detection_windows || []).slice(-1)[0] || {gate_open: status.last_gate_open})}`,
      }),
    ]);
  }

  function WaveformCard({item, index}) {
    const points = item.points || [];
    const polyline = points.map((value, pointIndex) => {
      const x = points.length === 1 ? 50 : (pointIndex / (points.length - 1)) * 100;
      const y = 36 - Math.max(-1, Math.min(1, Number(value))) * 34;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    }).join(" ");
    const thresholdY = dbfsToY(item.trigger_threshold_dbfs).toFixed(2);
    const outcome = item.outcome || (item.gate_open ? "detected" : "ignored");

    return h("div", {className: `waveform-card ${outcome !== "ignored" ? "gate-open" : ""}`}, [
      h("div", {key: "top", className: "waveform-meta"}, [
        h("span", {key: "index"}, `#${index}`),
        h("span", {key: "dbfs"}, `${item.dbfs} dBFS`),
        h("span", {key: "gate"}, windowOutcomeLabel(item)),
      ]),
      h("svg", {key: "svg", className: "waveform", viewBox: "0 0 100 72", preserveAspectRatio: "none"}, [
        h("line", {
          key: "threshold",
          className: "threshold-line",
          x1: "0",
          x2: "100",
          y1: thresholdY,
          y2: thresholdY,
        }),
        h("polyline", {key: "wave", className: "waveform-line", points: polyline}),
      ]),
      h("div", {key: "bottom", className: "waveform-meta"}, [
        h("span", {key: "floor"}, `底噪 ${item.noise_floor_dbfs} dBFS`),
        h("span", {key: "threshold"}, `阈值 ${item.trigger_threshold_dbfs} dBFS`),
      ]),
    ]);
  }

  function WaveformMonitor({status, autoRefresh, onAutoRefreshChange}) {
    const windows = status.recent_detection_windows || [];
    return h("section", {className: "monitor-panel"}, [
      h("div", {key: "header", className: "monitor-header"}, [
        h("div", {key: "title"}, [
          h("h2", {key: "h2", className: "monitor-title"}, "最近 10 个判定窗口"),
          h("div", {key: "hint", className: "muted"}, "阈值线按当前动态门限映射；刷新间隔与判定窗口一致。"),
        ]),
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
      windows.length
        ? h("div", {key: "grid", className: "waveform-grid"}, windows.map((item, index) => (
          h(WaveformCard, {key: index, item, index: index + 1})
        )))
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

  function EventTable({events}) {
    if (!events.length) {
      return h("div", {className: "empty"}, "暂无符合条件的事件。");
    }

    return h("table", null, [
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

    async function loadEvents(nextFilters = filters) {
      const query = buildQuery(nextFilters);
      const payload = await fetchJson(`/api/events${query ? `?${query}` : ""}`);
      setEvents(payload.events || []);
    }

    async function loadClassifications() {
      const payload = await fetchJson("/api/classifications");
      setClassifications(payload.classifications || []);
    }

    async function refresh(nextFilters = filters) {
      try {
        setError("");
        await Promise.all([loadStatus(), loadEvents(nextFilters)]);
      } catch (err) {
        setError(err.message);
      }
    }

    useEffect(() => {
      refresh();
      loadClassifications().catch((err) => setError(err.message));
    }, []);

    useEffect(() => {
      localStorage.setItem("audio-detect:auto-refresh", autoRefresh ? "1" : "0");
      if (!autoRefresh || !status.detection_window_seconds) {
        return undefined;
      }
      const timer = window.setInterval(() => {
        refresh();
      }, Number(status.detection_window_seconds) * 1000);
      return () => window.clearInterval(timer);
    }, [autoRefresh, status.detection_window_seconds, filters]);

    function applyFilters(nextFilters) {
      setFilters(nextFilters);
      loadEvents(nextFilters).then(() => setError("")).catch((err) => setError(err.message));
    }

    function resetFilters() {
      const emptyFilters = {classification: "", start_at: "", end_at: ""};
      setFilters(emptyFilters);
      loadEvents(emptyFilters).then(() => setError("")).catch((err) => setError(err.message));
    }

    return h(React.Fragment, null, [
      h("h1", {key: "title"}, "噪音监测面板"),
      h("p", {key: "intro"}, "系统会按固定检测窗口采集音频并运行分类，仅当窗口命中目标交通类标签时，才录制完整片段并写入事件记录。"),
      h(Tabs, {key: "tabs", activeTab, onChange: setActiveTab}),
      error ? h("div", {key: "error", className: "error"}, error) : null,
      activeTab === "status"
        ? h("section", {key: "status-tab", className: "tab-panel"}, [
          h(StatusCards, {key: "status", status}),
          status.last_error ? h("p", {key: "warning"}, `音频后端警告：${status.last_error}`) : null,
          h(WaveformMonitor, {
            key: "monitor",
            status,
            autoRefresh,
            onAutoRefreshChange: setAutoRefresh,
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
          h(EventTable, {key: "events", events}),
        ]),
    ]);
  }

  root.render(h(App));
})();
