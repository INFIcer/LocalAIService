/* LocalAIService 控制台 —— 纯原生 JS，无构建、无 CDN、不联网。 */
"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

const state = {
  view: "overview",
  autoRefresh: true,
  token: localStorage.getItem("localai_token") || "",
  authRequired: false,
  settings: null,
  settingsOriginal: {},
  downloadTask: null,
  downloadTimer: null,
  jobs: [],
  system: null,
  restartPolling: false,
  jobDetailOpen: false,
  jobDetailId: null,
  jobDetailData: null,
  jobDetailBlobs: [],
  jobDetailReturnFocus: null,
  lastJobInteraction: 0,
  copyPayloads: {},
};

const VIEW_META = {
  overview: ["概览", "服务状态、显存与任务队列一屏看完"],
  models: ["模型", "查看各能力权重、下载缺失文件、切换当前档位"],
  settings: ["设置", "按分组修改配置；保存后提示哪些需要重启"],
  jobs: ["任务", "任务队列、耗时与结果；排队中的任务可以取消"],
  system: ["系统", "开机自启、重启、日志与磁盘占用"],
};

/* ------------------------------------------------------------------ */
/* 基础工具                                                           */
/* ------------------------------------------------------------------ */
function escapeHtml(value) {
  return String(value == null ? "" : value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function fmtMb(value) {
  if (value == null || Number.isNaN(Number(value))) return "--";
  return `${Number(value).toLocaleString("zh-CN", { maximumFractionDigits: 1 })} MB`;
}

function fmtBytes(value) {
  if (value == null) return "--";
  const n = Number(value);
  if (!Number.isFinite(n)) return "--";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function fmtDuration(seconds) {
  if (seconds == null || Number.isNaN(Number(seconds))) return "--";
  const s = Math.max(0, Number(seconds));
  if (s < 1) return `${Math.round(s * 1000)} ms`;
  if (s < 60) return `${s.toFixed(1)} 秒`;
  if (s < 3600) return `${Math.floor(s / 60)} 分 ${Math.round(s % 60)} 秒`;
  return `${Math.floor(s / 3600)} 时 ${Math.round((s % 3600) / 60)} 分`;
}

function parseTime(value) {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

function timeAgo(value) {
  const date = parseTime(value);
  if (!date) return "--";
  const diff = (Date.now() - date.getTime()) / 1000;
  if (diff < 0) return date.toLocaleTimeString("zh-CN", { hour12: false });
  if (diff < 60) return `${Math.max(1, Math.round(diff))} 秒前`;
  if (diff < 3600) return `${Math.round(diff / 60)} 分钟前`;
  if (diff < 86400) return `${Math.round(diff / 3600)} 小时前`;
  return date.toLocaleString("zh-CN", { hour12: false });
}

function fmtClock(value) {
  const date = parseTime(value);
  if (!date) return "--";
  return date.toLocaleString("zh-CN", { hour12: false });
}

function statePill(value) {
  const text = String(value || "unknown");
  const map = {
    done: ["ok", "done"],
    running: ["info", "running"],
    queued: ["warn", "queued"],
    failed: ["bad", "failed"],
    canceled: ["muted", "canceled"],
    ok: ["ok", "ok"],
  };
  const [cls, label] = map[text] || ["muted", text];
  return `<span class="pill ${cls}">${escapeHtml(label)}</span>`;
}

function toast(message, type = "ok", ms = 4200) {
  const box = $("#toast");
  if (!box) return;
  box.textContent = message;
  box.className = `toast ${type}`;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => box.classList.add("hidden"), ms);
}

function showError(message) {
  const box = $("#error-banner");
  if (!box) return;
  box.textContent = message;
  box.classList.remove("hidden");
}
function clearError() {
  const box = $("#error-banner");
  if (box) box.classList.add("hidden");
}

function setConnection(ok, text) {
  const dot = $("#side-dot");
  const side = $("#side-status");
  const pill = $("#conn-pill");
  if (dot) dot.className = `dot ${ok ? "ok" : "bad"}`;
  if (side) side.textContent = text || (ok ? "服务在线" : "连接失败");
  if (pill) {
    pill.textContent = ok ? "在线" : "离线";
    pill.className = `pill ${ok ? "ok" : "bad"}`;
  }
}

/* ------------------------------------------------------------------ */
/* API                                                               */
/* ------------------------------------------------------------------ */
async function api(path, options = {}) {
  const headers = Object.assign({}, options.headers || {});
  if (options.body && typeof options.body === "string") {
    headers["Content-Type"] = "application/json";
  }
  if (state.token) headers["Authorization"] = `Bearer ${state.token}`;
  let response;
  try {
    response = await fetch(path, Object.assign({}, options, { headers }));
  } catch (err) {
    throw new Error(`网络错误：${err && err.message ? err.message : "请求失败"}`);
  }
  const raw = await response.text();
  let data = null;
  if (raw) {
    try {
      data = JSON.parse(raw);
    } catch (err) {
      data = raw;
    }
  }
  if (!response.ok) {
    if (response.status === 401) {
      state.authRequired = true;
      showTokenModal();
    }
    const detail =
      (data && data.detail) ||
      (data && data.error && data.error.message) ||
      (data && data.error) ||
      (typeof data === "string" && data) ||
      `HTTP ${response.status}`;
    const error = new Error(String(detail));
    error.status = response.status;
    error.data = data;
    throw error;
  }
  return data;
}

/* ------------------------------------------------------------------ */
/* 导航                                                               */
/* ------------------------------------------------------------------ */
function switchView(view) {
  if (!VIEW_META[view]) return;
  state.view = view;
  $$(".nav-item").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.view === view);
  });
  $$(".view").forEach((section) => {
    section.classList.toggle("active", section.id === `view-${view}`);
  });
  const [title, sub] = VIEW_META[view];
  $("#view-title").textContent = title;
  $("#view-sub").textContent = sub;
  loadView(view, false);
}

async function loadView(view, quiet) {
  clearError();
  try {
    if (view === "overview") await refreshOverview(quiet);
    if (view === "models") await loadModels();
    if (view === "settings") await loadSettings();
    if (view === "jobs") await loadJobs();
    if (view === "system") await loadSystem();
  } catch (err) {
    if (!(err.status === 401 && !state.token)) showError(err.message || String(err));
    if (!quiet) toast(err.message || String(err), "error");
  }
}

/* ------------------------------------------------------------------ */
/* 概览                                                               */
/* ------------------------------------------------------------------ */
const CAP_ORDER = ["tts", "asr", "whisper", "translate", "speaker", "separate", "media"];
const CAP_TITLES = {
  tts: "语音合成 TTS",
  asr: "语音识别 SenseVoice",
  whisper: "语音识别 whisper",
  translate: "翻译 NLLB",
  speaker: "声纹 campplus",
  separate: "人声分离 Demucs",
  media: "媒体 ffmpeg",
};

async function refreshOverview(quiet = false) {
  const [caps, health, vram, jobs, logs] = await Promise.all([
    api("/capabilities").catch((err) => {
      if (err.status === 401) throw err;
      return null;
    }),
    api("/health").catch(() => null),
    api("/vram").catch(() => null),
    api("/jobs?limit=10").catch(() => null),
    api("/admin/logs?lines=8").catch(() => null),
  ]);

  if (caps) {
    state.authRequired = Boolean(caps.auth && caps.auth.required);
    updateHostWarning(caps.listen && caps.listen.host);
    renderCapabilities(caps, vram);
    renderGpu(vram, caps);
  }
  if (health) renderJobStats(health.jobs || {});
  renderRecentJobs(jobs);
  renderRecentLogs(logs);
  setConnection(true, "服务在线");
  if (!quiet) clearError();
}

function renderGpu(vram, caps) {
  const total = vram && Number(vram.total_mb) > 0 ? Number(vram.total_mb) : 0;
  const free = vram ? Number(vram.free_mb) || 0 : 0;
  const margin = vram && vram.margin_mb != null ? Number(vram.margin_mb) : (caps && caps.vram ? caps.vram.margin_mb : null);
  const badge = $("#gpu-state");
  if (!total) {
    $("#gpu-used").textContent = "--";
    $("#gpu-total").textContent = "/ 不可用";
    $("#gpu-bar").style.width = "0%";
    $("#gpu-free").textContent = "未检测到 GPU";
    $("#gpu-margin").textContent = margin == null ? "" : `安全余量 ${margin} MB`;
    if (badge) {
      badge.textContent = "无 GPU / 驱动未就绪";
      badge.className = "pill muted";
    }
    return;
  }
  const used = Math.max(0, total - free);
  const percent = Math.min(100, Math.max(0, (used / total) * 100));
  $("#gpu-used").textContent = used.toLocaleString("zh-CN");
  $("#gpu-total").textContent = `/ ${total.toLocaleString("zh-CN")} MB`;
  const bar = $("#gpu-bar");
  bar.style.width = `${percent.toFixed(1)}%`;
  bar.className = `progress-fill${percent >= 92 ? " crit" : percent >= 80 ? " hot" : ""}`;
  $("#gpu-free").textContent = `空闲 ${free.toLocaleString("zh-CN")} MB`;
  $("#gpu-margin").textContent = margin == null ? "" : `安全余量 ${margin} MB`;
  if (badge) {
    badge.textContent = `已用 ${percent.toFixed(0)}%`;
    badge.className = `pill ${percent >= 92 ? "bad" : percent >= 80 ? "warn" : "ok"}`;
  }
}

function renderCapabilities(caps, vram) {
  const grid = $("#cap-grid");
  const capabilities = (caps && caps.capabilities) || {};
  const vramModels = {};
  ((vram && vram.models) || []).forEach((item) => {
    vramModels[item.name] = item;
  });

  const html = CAP_ORDER.filter((name) => capabilities[name]).map((name) => {
    const item = capabilities[name] || {};
    const loaded = Boolean(item.loaded);
    const ready = Boolean(item.ready);
    const estimate = vramModels[name] && vramModels[name].estimate_mb != null
      ? vramModels[name].estimate_mb
      : item.estimate_mb;
    const idle = item.idle_unload_seconds;
    const endpoints = Array.isArray(item.endpoints) ? item.endpoints.length : 0;
    const title = CAP_TITLES[name] || item.title || name;
    const model = item.model || item.detail || "—";
    return `
      <div class="cap-card ${loaded ? "loaded" : ""}">
        <div class="cap-top">
          <span class="cap-title">${escapeHtml(title)}</span>
          <span class="pill ${loaded ? "ok" : ready ? "info" : "warn"}">${loaded ? "已加载" : ready ? "就绪" : "未就绪"}</span>
        </div>
        <div class="cap-model" title="${escapeHtml(item.model_dir || "")}">${escapeHtml(model)}</div>
        <div class="cap-meta">
          <span>显存约 ${escapeHtml(fmtMb(estimate))}</span>
          <span>${idle == null ? "空闲卸载关闭" : `空闲 ${idle}s 卸载`}</span>
          <span>${endpoints} 个端点</span>
        </div>
      </div>`;
  }).join("");
  grid.innerHTML = html || '<div class="empty">没有能力信息</div>';
}

function renderJobStats(jobs) {
  const box = $("#job-stat-grid");
  const items = [
    ["排队", jobs.queued || 0, "warn"],
    ["运行", jobs.running || 0, "info"],
    ["完成", jobs.done || 0, "ok"],
    ["失败", jobs.failed || 0, "bad"],
  ];
  box.innerHTML = items.map(([label, value, cls]) => `
    <div class="stat-card">
      <span class="stat-value">${escapeHtml(value)}</span>
      <span class="stat-label"><span class="dot ${cls === "ok" ? "ok" : cls === "bad" ? "bad" : ""}"></span> ${escapeHtml(label)}</span>
    </div>`).join("");
}

function renderRecentJobs(data) {
  const tbody = $("#recent-jobs");
  if (!data || !Array.isArray(data.jobs) || !data.jobs.length) {
    tbody.innerHTML = '<tr><td colspan="4" class="empty">暂无任务</td></tr>';
    return;
  }
  tbody.innerHTML = data.jobs.slice(0, 10).map((job) => {
    const started = parseTime(job.started_at);
    const finished = parseTime(job.finished_at);
    const now = Date.now();
    let elapsed = "--";
    if (started && finished) elapsed = fmtDuration((finished - started) / 1000);
    else if (started) elapsed = fmtDuration((now - started.getTime()) / 1000);
    return `
      <tr class="clickable" data-job-id="${escapeHtml(job.id)}" tabindex="0" role="button"
          aria-label="任务 ${escapeHtml(job.id)}，按回车或单击查看详情">
        <td title="${escapeHtml(job.id)}">${escapeHtml(timeAgo(job.created_at))}</td>
        <td>${escapeHtml(job.kind || "")}</td>
        <td>${statePill(job.state)}</td>
        <td>${escapeHtml(elapsed)}</td>
      </tr>`;
  }).join("");
  $$("#recent-jobs tr[data-job-id]").forEach((row) => {
    const jobId = row.dataset.jobId;
    row.addEventListener("click", () => {
      switchView("jobs");
      openJobDetail(jobId);
    });
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        switchView("jobs");
        openJobDetail(jobId);
      }
    });
  });
}

function renderRecentLogs(log) {
  const box = $("#recent-logs");
  if (!log) {
    box.textContent = "（需要令牌或日志不可读）";
    return;
  }
  if (log.exists === false) {
    box.textContent = log.detail || "日志文件还不存在";
    return;
  }
  box.textContent = (log.text || "").trim() || "（日志为空）";
  box.scrollTop = box.scrollHeight;
}

/* ------------------------------------------------------------------ */
/* 模型                                                               */
/* ------------------------------------------------------------------ */
async function loadModels() {
  const data = await api("/admin/models");
  state.models = data.models || [];
  renderModels();
}

function modelActionButtons(item) {
  const cap = item.capability;
  const name = item.name;
  if (cap === "download") {
    return '<span class="hint">由 engine/.venv 的 pip 管理</span>';
  }
  const canActivate = cap === "whisper" || cap === "asr";
  const downloadLabel = item.present ? "重新下载" : "下载";
  const buttons = [
    `<button class="small" data-action="download" data-cap="${escapeHtml(cap)}" data-name="${escapeHtml(name)}">${downloadLabel}</button>`,
  ];
  if (canActivate) {
    const activateName = cap === "asr" ? "sensevoice" : name;
    buttons.push(
      `<button class="small ${item.active ? "ghost" : "primary"}" data-action="activate" data-cap="${escapeHtml(cap)}" data-name="${escapeHtml(activateName)}" ${item.active ? "disabled" : ""}>${item.active ? "当前档位" : "切换"}</button>`
    );
  }
  return `<div class="actions">${buttons.join("")}</div>`;
}

function renderModels() {
  const container = $("#models-container");
  const models = state.models || [];
  if (!models.length) {
    container.innerHTML = '<div class="empty">模型清单为空</div>';
    return;
  }
  const groups = [];
  const byCap = new Map();
  models.forEach((item) => {
    if (!byCap.has(item.capability)) {
      byCap.set(item.capability, []);
      groups.push(item.capability);
    }
    byCap.get(item.capability).push(item);
  });
  const totalMissing = models.filter((item) => !item.present).length;
  $("#models-summary").textContent = `共 ${models.length} 项模型，${totalMissing} 项缺少权重；当前档位用「当前档位」标记。`;

  container.innerHTML = groups.map((cap) => {
    const list = byCap.get(cap);
    const title = (list[0] && list[0].capability_title) || cap;
    const rows = list.map((item) => {
      const missing = (item.missing_files || []).length
        ? `<div class="missing">缺：${escapeHtml(item.missing_files.join("、"))}</div>`
        : "";
      return `
        <tr>
          <td>
            <div class="model-name">${escapeHtml(item.name)}</div>
            <div class="path">${escapeHtml(item.path)}</div>
          </td>
          <td><span class="pill ${item.present ? "ok" : "warn"}">${item.present ? "已下载" : "未下载"}</span></td>
          <td>${escapeHtml(fmtMb(item.size_mb))}</td>
          <td>${item.active ? '<span class="pill info">当前档位</span>' : '<span class="faint">—</span>'}</td>
          <td>${missing}${item.note ? `<div class="hint">${escapeHtml(item.note)}</div>` : ""}</td>
          <td>${modelActionButtons(item)}</td>
        </tr>`;
    }).join("");
    return `
      <div class="panel cap-group">
        <div class="cap-group-head"><h3>${escapeHtml(title)}</h3><span class="pill muted">${escapeHtml(cap)}</span></div>
        <div class="table-wrap"><table class="model-table">
          <thead><tr><th>模型</th><th>状态</th><th>大小</th><th>档位</th><th>缺失 / 说明</th><th>操作</th></tr></thead>
          <tbody>${rows}</tbody>
        </table></div>
      </div>`;
  }).join("");

  $$("#models-container [data-action]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const cap = btn.dataset.cap;
      const name = btn.dataset.name;
      if (btn.dataset.action === "download") startDownload(cap, name);
      if (btn.dataset.action === "activate") activateModel(cap, name);
    });
  });
}

async function startDownload(capability, name) {
  if (state.downloadTask) {
    toast("已有下载任务在运行，请等它结束", "error");
    return;
  }
  try {
    const data = await api("/admin/models/download", {
      method: "POST",
      body: JSON.stringify({ capability, name }),
    });
    state.downloadTask = data.task_id;
    showDownloadBanner(`下载已启动：${capability} / ${name}（任务 ${data.task_id}）`, "info");
    pollDownload();
  } catch (err) {
    toast(err.message || String(err), "error");
  }
}

function showDownloadBanner(text, type = "info") {
  const box = $("#download-task");
  box.className = `banner ${type}`;
  box.innerHTML = text;
  box.classList.remove("hidden");
}

async function pollDownload() {
  clearTimeout(state.downloadTimer);
  if (!state.downloadTask) return;
  try {
    const data = await api(`/admin/models/tasks/${encodeURIComponent(state.downloadTask)}`);
    const percent = data.percent == null ? "" : `（${data.percent}%）`;
    showDownloadBanner(
      `下载中：${escapeHtml(data.capability)} / ${escapeHtml(data.name)} ${percent} — 已写入 ${escapeHtml(fmtMb(data.downloaded_mb))}，目标目录 ${escapeHtml(fmtMb(data.size_mb))}` +
      '<div class="progress task-progress"><div class="progress-fill task-indeterminate"></div></div>',
      "info"
    );
    if (data.done) {
      showDownloadBanner(`✅ ${escapeHtml(data.capability)} / ${escapeHtml(data.name)} 下载完成（${escapeHtml(fmtMb(data.size_mb))}）`, "ok");
      toast("模型下载完成", "ok");
      state.downloadTask = null;
      await loadModels();
      return;
    }
    if (data.failed) {
      showDownloadBanner(`❌ 下载失败：${escapeHtml(data.error || "未知错误")}`, "error");
      toast("模型下载失败", "error");
      state.downloadTask = null;
      await loadModels();
      return;
    }
    state.downloadTimer = setTimeout(pollDownload, 1000);
  } catch (err) {
    state.downloadTask = null;
    showDownloadBanner(`下载任务查询失败：${escapeHtml(err.message || String(err))}`, "error");
  }
}

async function activateModel(capability, name) {
  try {
    const data = await api("/admin/models/activate", {
      method: "POST",
      body: JSON.stringify({ capability, name }),
    });
    toast(data.note || "已切换", "ok");
    if (data.warnings && data.warnings.length) {
      toast(data.warnings.join("；"), "error", 7000);
    }
    await loadModels();
  } catch (err) {
    toast(err.message || String(err), "error");
  }
}

/* ------------------------------------------------------------------ */
/* 设置                                                               */
/* ------------------------------------------------------------------ */
async function loadSettings() {
  const data = await api("/admin/settings");
  state.settings = data.settings || {};
  state.settingsOriginal = {};
  Object.entries(state.settings).forEach(([key, item]) => {
    state.settingsOriginal[key] = item.value;
  });
  renderSettings(data);
}

function settingTags(item) {
  const tags = [];
  const labels = { env: "来源 env", file: "来源 file", default: "来源 default" };
  tags.push(`<span class="tag ${item.source}">${labels[item.source] || item.source}</span>`);
  if (item.requires_restart) tags.push('<span class="tag restart">需重启</span>');
  if (item.default !== null && item.default !== undefined && item.default !== "" && item.type !== "path") {
    const def = item.type === "bool" ? (item.default ? "true" : "false") : item.default;
    tags.push(`<span class="tag default">默认 ${escapeHtml(def)}</span>`);
  }
  if (item.secret && item.enabled) tags.push('<span class="tag file">已设置</span>');
  return tags.join("");
}

function settingControl(item) {
  const key = escapeHtml(item.key);
  const value = item.value;
  if (item.secret) {
    return `<input data-setting-key="${key}" type="password" autocomplete="new-password" value="" placeholder="${item.enabled ? "已设置（留空不修改）" : "留空表示不鉴权"}">`;
  }
  if (item.type === "bool") {
    return `<select data-setting-key="${key}">
      <option value="true" ${value ? "selected" : ""}>开启</option>
      <option value="false" ${!value ? "selected" : ""}>关闭</option>
    </select>`;
  }
  if (item.type === "choice") {
    return `<select data-setting-key="${key}">${(item.choices || []).map((choice) =>
      `<option value="${escapeHtml(choice)}" ${String(choice) === String(value) ? "selected" : ""}>${escapeHtml(choice)}</option>`
    ).join("")}</select>`;
  }
  if (item.type === "int") {
    return `<input data-setting-key="${key}" type="number" value="${escapeHtml(value)}" min="${item.min != null ? item.min : 0}" max="${item.max != null ? item.max : ""}" step="1">`;
  }
  return `<input data-setting-key="${key}" type="text" value="${escapeHtml(value == null ? "" : value)}" spellcheck="false">`;
}

function renderSettings(data) {
  const container = $("#settings-container");
  const groups = data.groups || [];
  container.innerHTML = groups.map((group) => {
    const rows = group.items.map((item) => `
      <div class="setting-row" data-row-key="${escapeHtml(item.key)}">
        <div>
          <div class="setting-title">${escapeHtml(item.title)} <span class="faint">${escapeHtml(item.key)}</span></div>
          <div class="setting-desc">${escapeHtml(item.description || "")}</div>
          <div class="setting-meta">${settingTags(item)}</div>
        </div>
        <div class="setting-control">${settingControl(item)}</div>
        <div class="hint">${item.source === "env" ? "环境变量优先" : item.requires_restart ? "重启后生效" : "保存即生效"}</div>
      </div>`).join("");
    return `<div class="panel settings-group">
      <div class="panel-head"><h2>${escapeHtml(group.title)}</h2><span class="pill muted">${escapeHtml(group.group)}</span></div>
      ${rows}
    </div>`;
  }).join("");

  $$("#settings-container [data-setting-key]").forEach((input) => {
    input.addEventListener("input", () => markSettingDirty(input));
    input.addEventListener("change", () => markSettingDirty(input));
  });
  $("#settings-result").classList.add("hidden");
}

function markSettingDirty(input) {
  const key = input.dataset.settingKey;
  const row = input.closest(".setting-row");
  const item = state.settings[key];
  if (!item || !row) return;
  let current = input.value;
  if (item.type === "bool") current = current === "true";
  const original = state.settingsOriginal[key];
  const same = item.secret ? current === "" : String(current) === String(original == null ? "" : original);
  row.classList.toggle("changed", !same);
  let tag = row.querySelector(".tag.dirty");
  if (!same) {
    if (!tag) {
      tag = document.createElement("span");
      tag.className = "tag dirty";
      tag.textContent = "已修改";
      row.querySelector(".setting-meta").appendChild(tag);
    }
  } else if (tag) {
    tag.remove();
  }
}

async function saveSettings() {
  if (!state.settings) return;
  const values = {};
  $$("#settings-container [data-setting-key]").forEach((input) => {
    const key = input.dataset.settingKey;
    const item = state.settings[key];
    if (!item) return;
    let raw = input.value;
    if (item.secret) {
      if (!raw) return;
    } else if (item.type === "bool") {
      raw = raw === "true";
    } else if (item.type === "int") {
      if (raw === "") return;
      raw = Number(raw);
    }
    const original = state.settingsOriginal[key];
    if (!item.secret && String(raw) === String(original == null ? "" : original)) return;
    values[key] = raw;
  });

  const keys = Object.keys(values);
  if (!keys.length) {
    toast("没有改动需要保存", "error");
    return;
  }
  const button = $("#settings-save");
  button.disabled = true;
  button.textContent = "保存中…";
  try {
    const result = await api("/admin/settings", {
      method: "PUT",
      body: JSON.stringify({ values }),
    });
    const resultBox = $("#settings-result");
    const parts = [`✅ 已保存 ${result.applied.length} 项：${result.applied.join("、")}`];
    if (result.requires_restart && result.requires_restart.length) {
      parts.push(`⚠️ 需要重启才真正生效：${result.requires_restart.join("、")}`);
    }
    if (result.warnings && result.warnings.length) {
      parts.push(...result.warnings.map((item) => `⚠️ ${item}`));
    }
    resultBox.className = `banner ${result.requires_restart && result.requires_restart.length ? "warn" : "ok"}`;
    resultBox.textContent = parts.join("\n");
    resultBox.style.whiteSpace = "pre-wrap";
    resultBox.classList.remove("hidden");
    toast("设置已保存", "ok");

    if (Object.prototype.hasOwnProperty.call(values, "token") && !(result.overridden_by_env || []).includes("token")) {
      state.token = String(values.token || "");
      if (state.token) localStorage.setItem("localai_token", state.token);
      else localStorage.removeItem("localai_token");
    }
    await loadSettings();
    resultBox.className = `banner ${result.requires_restart && result.requires_restart.length ? "warn" : "ok"}`;
    resultBox.textContent = parts.join("\n");
    resultBox.classList.remove("hidden");
  } catch (err) {
    toast(err.message || String(err), "error", 7000);
    showError(err.message || String(err));
  } finally {
    button.disabled = false;
    button.textContent = "保存设置";
  }
}

/* ------------------------------------------------------------------ */
/* 任务                                                               */
/* ------------------------------------------------------------------ */
async function loadJobs() {
  const filter = $("#jobs-filter") ? $("#jobs-filter").value : "";
  const query = new URLSearchParams({ limit: "200" });
  if (filter) query.set("state", filter);
  const data = await api(`/jobs?${query.toString()}`);
  state.jobs = data.jobs || [];
  const tbody = $("#jobs-tbody");
  $("#jobs-summary").textContent = `共 ${data.total} 条${filter ? `（筛选 ${filter}）` : ""}，显示最近 ${state.jobs.length} 条`;
  if (!state.jobs.length) {
    tbody.innerHTML = '<tr><td colspan="7" class="empty">没有任务</td></tr>';
    return;
  }
  const now = Date.now();
  tbody.innerHTML = state.jobs.map((job) => {
    const created = parseTime(job.created_at);
    const started = parseTime(job.started_at);
    const finished = parseTime(job.finished_at);
    let wait = "--";
    if (created && started) wait = fmtDuration((started - created) / 1000);
    else if (created && job.state === "queued") wait = fmtDuration((now - created.getTime()) / 1000);
    let duration = "--";
    if (started && finished) duration = fmtDuration((finished - started) / 1000);
    else if (started) duration = fmtDuration((now - started.getTime()) / 1000);
    const terminal = ["done", "failed", "canceled"].includes(job.state);
    return `
      <tr class="clickable job-row" data-job-id="${escapeHtml(job.id)}" tabindex="0"
          role="button" aria-label="任务 ${escapeHtml(job.id)}，按回车或双击查看详情">
        <td title="${escapeHtml(job.id)}">${escapeHtml(fmtClock(job.created_at))}</td>
        <td>${escapeHtml(job.kind)}<div class="path">${escapeHtml(job.action || "")}</div></td>
        <td>${statePill(job.state)}</td>
        <td>${escapeHtml(job.priority == null ? 0 : job.priority)}</td>
        <td>${escapeHtml(wait)}</td>
        <td>${escapeHtml(duration)}</td>
        <td>
          <div class="actions">
            <button class="small ghost" data-detail="${escapeHtml(job.id)}" type="button">详情</button>
            ${terminal ? '<span class="faint">—</span>' : `<button class="small danger" data-cancel="${escapeHtml(job.id)}" type="button">取消</button>`}
          </div>
        </td>
      </tr>`;
  }).join("");

  $$("#jobs-tbody tr[data-job-id]").forEach((row) => {
    const jobId = row.dataset.jobId;
    row.addEventListener("dblclick", () => openJobDetail(jobId));
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openJobDetail(jobId);
      }
    });
  });
  $$("#jobs-tbody [data-detail]").forEach((btn) => {
    btn.addEventListener("click", (event) => {
      event.stopPropagation();
      openJobDetail(btn.dataset.detail);
    });
  });
  $$("#jobs-tbody [data-cancel]").forEach((btn) => {
    btn.addEventListener("click", (event) => {
      event.stopPropagation();
      cancelJob(btn.dataset.cancel);
    });
  });
}

/* ------------------------------------------------------------------ */
/* 任务详情弹窗                                                       */
/* ------------------------------------------------------------------ */
const AUDIO_RE = /\.(wav|mp3|m4a|flac|ogg|opus|aac)$/i;
const IMAGE_RE = /\.(png|jpe?g|webp|gif|bmp)$/i;
const TEXT_KEY_RE = /(text|prompt|subtitle|ocr|transcript|content|query|note|description|summary)/i;

function fmtMaybe(value) {
  return value == null || value === "" ? "--" : String(value);
}

function copyButton(text, label = "复制") {
  const key = `copy-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
  state.copyPayloads[key] = String(text == null ? "" : text);
  return `<button class="small ghost" type="button" data-copy-key="${key}">${escapeHtml(label)}</button>`;
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    toast("已复制到剪贴板", "ok", 1600);
    return;
  } catch (err) {
    // 非 https / 旧浏览器没有 clipboard API，退回隐藏 textarea。
  }
  const area = document.createElement("textarea");
  area.value = text;
  area.style.position = "fixed";
  area.style.opacity = "0";
  document.body.appendChild(area);
  area.select();
  try {
    document.execCommand("copy");
    toast("已复制到剪贴板", "ok", 1600);
  } catch (err) {
    toast("复制失败，请手动选中文本", "error");
  } finally {
    area.remove();
  }
}

function jsonBlock(value, emptyText = "（空）") {
  if (value == null || (typeof value === "object" && !Object.keys(value).length)) {
    return `<div class="empty">${escapeHtml(emptyText)}</div>`;
  }
  const text = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  return `<pre class="log-box detail-json">${escapeHtml(text)}</pre>`;
}

function collectTextFields(obj, prefix = "", out = []) {
  if (!obj || typeof obj !== "object" || out.length >= 12) return out;
  Object.entries(obj).forEach(([key, value]) => {
    const full = prefix ? `${prefix}.${key}` : key;
    if (typeof value === "string" && value.trim() && TEXT_KEY_RE.test(key)) {
      out.push([full, value]);
    } else if (value && typeof value === "object" && !Array.isArray(value) && prefix.split(".").length < 2) {
      collectTextFields(value, full, out);
    }
  });
  return out;
}

function renderTextFields(payload) {
  const fields = collectTextFields(payload);
  if (!fields.length) return "";
  return `<div class="detail-texts">${fields.map(([key, value]) => `
    <div class="detail-text-field">
      <div class="detail-text-label">${escapeHtml(key)} ${copyButton(value, "复制原文")}</div>
      <pre class="detail-text">${escapeHtml(value)}</pre>
    </div>`).join("")}</div>`;
}

function looksLikePath(text) {
  return typeof text === "string" &&
    text.length > 1 &&
    !/[\r\n]/.test(text) &&
    (/^[A-Za-z]:[\\/]/.test(text) || text.startsWith("\\\\") || text.startsWith("/"));
}

function collectResultPaths(value, out = [], depth = 0) {
  if (depth > 6 || value == null) return out;
  if (typeof value === "string") {
    if (looksLikePath(value) && !out.some((item) => item.path === value)) {
      out.push({ path: value });
    }
    return out;
  }
  if (Array.isArray(value)) {
    value.forEach((item) => collectResultPaths(item, out, depth + 1));
    return out;
  }
  if (typeof value === "object") {
    Object.values(value).forEach((item) => collectResultPaths(item, out, depth + 1));
  }
  return out;
}

function primaryResultPath(job) {
  const result = (job && job.result) || {};
  for (const key of ["path", "output_path", "output", "audio_path", "file"]) {
    if (typeof result[key] === "string" && result[key]) return result[key];
  }
  return "";
}

function renderResultFiles(job) {
  const paths = collectResultPaths(job.result);
  if (!paths.length) {
    return '<div class="hint">结果里没有可识别的本地文件路径。</div>';
  }
  const primary = primaryResultPath(job);
  return `<div class="detail-files">${paths.map(({ path }) => {
    const isPrimary = primary && path === primary;
    let tag = "文件";
    let media = '<div class="hint">此类型不内嵌预览，可复制路径后在本地打开。</div>';
    if (AUDIO_RE.test(path)) {
      tag = "音频";
      media = isPrimary
        ? '<audio class="detail-audio" controls preload="none" data-primary-audio></audio><div class="hint" data-media-note>正在加载音频…</div>'
        : '<div class="hint">该音频不是任务主结果，无法直接串流；可复制路径在本地播放。</div>';
    } else if (IMAGE_RE.test(path)) {
      tag = "图片";
      if (isPrimary) {
        media = '<img class="detail-thumb" alt="任务结果图片" data-primary-image hidden><div class="hint" data-media-note>正在加载缩略图…</div>';
      } else {
        media = `<img class="detail-thumb" alt="结果图片" data-frame-path="${escapeHtml(path)}" hidden><div class="hint" data-media-note>正在加载缩略图…</div>`;
      }
    }
    return `<div class="detail-file">
      <div class="detail-file-head"><span class="tag">${tag}</span><span class="detail-file-path mono">${escapeHtml(path)}</span></div>
      ${media}
      <div class="actions">${copyButton(path, "复制路径")}</div>
    </div>`;
  }).join("")}</div>`;
}

async function fetchBlob(path) {
  const headers = {};
  if (state.token) headers.Authorization = `Bearer ${state.token}`;
  const response = await fetch(path, { headers, cache: "no-store" });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      detail = (body && (body.detail || body.error)) || detail;
    } catch (err) {
      // 非 JSON（例如 401 空响应）直接用状态码。
    }
    throw new Error(String(detail));
  }
  return response.blob();
}

function trackBlobUrl(blob) {
  const url = URL.createObjectURL(blob);
  state.jobDetailBlobs.push(url);
  return url;
}

function revokeJobBlobs() {
  (state.jobDetailBlobs || []).forEach((url) => {
    try { URL.revokeObjectURL(url); } catch (err) { /* 已释放则忽略 */ }
  });
  state.jobDetailBlobs = [];
}

async function attachJobMedia(job) {
  revokeJobBlobs();
  const body = $("#job-modal-body");
  if (!body) return;
  const resultEndpoint = `/jobs/${encodeURIComponent(job.id)}/result`;
  const primary = primaryResultPath(job);

  const audio = $("[data-primary-audio]", body);
  if (audio && primary && AUDIO_RE.test(primary)) {
    const note = audio.parentElement ? audio.parentElement.querySelector("[data-media-note]") : null;
    try {
      const blob = await fetchBlob(resultEndpoint);
      if (blob.type.startsWith("audio/")) {
        audio.src = trackBlobUrl(blob);
        if (note) note.textContent = `${blob.type} · ${fmtBytes(blob.size)}`;
      } else if (note) {
        note.textContent = "该接口没有返回音频（可能是任务结果 JSON），请复制路径在本地播放。";
      }
    } catch (err) {
      if (note) note.textContent = `加载音频失败：${err.message || String(err)}`;
    }
  }

  const images = $$("img[data-primary-image], img[data-frame-path]", body);
  for (const img of images) {
    const note = img.parentElement ? img.parentElement.querySelector("[data-media-note]") : null;
    const fromFrame = img.dataset.framePath;
    const requests = [];
    if (!fromFrame) requests.push(resultEndpoint);
    if (fromFrame || !primary || !AUDIO_RE.test(primary)) {
      const path = fromFrame || primary;
      if (path) requests.push(`/vision/frame?path=${encodeURIComponent(path)}`);
    }
    let loaded = false;
    for (const request of requests) {
      try {
        const blob = await fetchBlob(request);
        if (!blob.type.startsWith("image/")) continue;
        img.src = trackBlobUrl(blob);
        img.hidden = false;
        if (note) note.textContent = `${blob.type} · ${fmtBytes(blob.size)}`;
        loaded = true;
        break;
      } catch (err) {
        // 换下一个数据源；都失败再显示提示。
      }
    }
    if (!loaded && note) {
      note.textContent = "缩略图不可用（路径不在可预览目录或文件已删除）；可复制路径后在本地打开。";
    }
  }
}

async function loadJobDetail(jobId) {
  let detail = null;
  let loadError = "";
  try {
    detail = await api(`/jobs/${encodeURIComponent(jobId)}`);
  } catch (err) {
    loadError = err.message || String(err);
  }
  if (!state.jobDetailOpen || state.jobDetailId !== jobId) return;
  const fallback = (state.jobs || []).find((job) => job.id === jobId) || null;
  if (!detail && !fallback) {
    renderJobDetailError(loadError || "任务不存在");
    return;
  }
  // 列表行里已经有完整记录；详情接口失败时用它兜底，成功时以详情为准。
  const job = Object.assign({}, fallback || {}, detail || {});
  state.jobDetailData = job;
  renderJobDetail(job, loadError);
}

function waitDuration(job) {
  const created = parseTime(job.created_at);
  const started = parseTime(job.started_at);
  if (created && started) return fmtDuration((started - created) / 1000);
  if (created && job.state === "queued") {
    return `${fmtDuration((Date.now() - created.getTime()) / 1000)}（排队中）`;
  }
  return "--";
}

function runDuration(job) {
  const started = parseTime(job.started_at);
  const finished = parseTime(job.finished_at);
  if (started && finished) return fmtDuration((finished - started) / 1000);
  if (started && job.state === "running") {
    return `${fmtDuration((Date.now() - started.getTime()) / 1000)}（进行中）`;
  }
  return "--";
}

function renderTimeline(job) {
  const stateText = String(job.state || "unknown");
  const steps = [
    ["创建", job.created_at],
    ["开始执行", job.started_at],
    ["结束", job.finished_at],
  ];
  const items = steps.map(([label, value], index) => {
    const date = parseTime(value);
    let text = date ? fmtClock(value) : "—";
    if (!date) {
      if (stateText === "queued" && index > 0) text = "尚未开始（排队中）";
      else if (stateText === "running" && index === 2) text = "进行中…";
      else if (stateText === "canceled") text = "已取消";
    }
    const cls = date ? "done" : index === 0 && stateText !== "queued" ? "done" : "pending";
    return `<div class="timeline-item ${cls}">
      <span class="timeline-dot"></span>
      <span class="timeline-label">${escapeHtml(label)}</span>
      <span class="timeline-time mono">${escapeHtml(text)}</span>
    </div>`;
  }).join("");

  const errorText = String(job.error || "");
  const notes = [];
  if (stateText === "canceled") {
    notes.push("该任务已取消：如果是主动取消，error 为空是正常现象；有 error 时说明执行器报错后任务被终止。");
  }
  if (/超时|timeout|timed out/i.test(errorText)) {
    notes.push("错误信息里出现了「超时 / timeout」，这次失败是执行超时，不是测试用例产生的。");
  }
  if (stateText === "running" && (job.meta || {}).cancel_requested) {
    notes.push("已请求取消，正在等待执行器退出（running 状态可能还会持续几秒）。");
  }
  return `<div class="detail-section">
    <h3 class="detail-h">时间线</h3>
    <div class="timeline">${items}</div>
    ${notes.map((note) => `<div class="detail-note">${escapeHtml(note)}</div>`).join("")}
  </div>`;
}

function renderJobDetail(job, loadError) {
  const box = $("#job-modal-body");
  if (!box) return;
  const stateText = String(job.state || "unknown");
  const terminal = ["done", "failed", "canceled"].includes(stateText);
  const errorText = String(job.error || "");
  const timeout = /超时|timeout|timed out/i.test(errorText);

  $("#job-modal-sub").textContent = [job.kind, job.action].filter(Boolean).join(" · ") || "任务";
  $("#job-modal-meta").textContent = `任务 ${job.id}`;
  const cancelBtn = $("#job-modal-cancel");
  cancelBtn.disabled = terminal;
  cancelBtn.classList.toggle("hidden", terminal);
  if (!terminal) cancelBtn.textContent = stateText === "running" ? "请求取消" : "取消任务";

  const summary = [
    ["任务 ID", `<span class="mono">${escapeHtml(job.id || "--")}</span> ${copyButton(job.id || "", "复制 ID")}`],
    ["类型 / 动作", `${escapeHtml(job.kind || "--")} / <span class="mono">${escapeHtml(job.action || "--")}</span>`],
    ["状态", `${statePill(job.state)}${timeout ? ' <span class="pill bad">超时</span>' : ""}`],
    ["优先级", escapeHtml(fmtMaybe(job.priority))],
    ["创建时间", escapeHtml(job.created_at ? fmtClock(job.created_at) : "--")],
    ["开始时间", escapeHtml(job.started_at ? fmtClock(job.started_at) : "--")],
    ["结束时间", escapeHtml(job.finished_at ? fmtClock(job.finished_at) : "--")],
    ["等待时长", escapeHtml(waitDuration(job))],
    ["执行时长", escapeHtml(runDuration(job))],
    ["尝试次数", escapeHtml(fmtMaybe(job.attempts))],
    ["客户端 IP", escapeHtml(job.client_ip || "--")],
  ];
  if ((job.meta || {}).cancel_requested) summary.push(["取消请求", "已发出（等待执行器退出）"]);

  const body = [
    loadError ? `<div class="banner error">读取完整详情失败：${escapeHtml(loadError)}；下面用任务列表里的缓存数据展示，可点「刷新详情」重试。</div>` : "",
    `<div class="detail-section">
      <h3 class="detail-h">概要</h3>
      <div class="detail-grid">${summary.map(([label, value], index) => `
        <div class="detail-kv${index === 0 ? " detail-kv-wide" : ""}"><span class="detail-label">${escapeHtml(label)}</span><span class="detail-value">${value}</span></div>`).join("")}
      </div>
    </div>`,
    `<div class="detail-section">
      <h3 class="detail-h">请求参数 <span class="hint">payload</span></h3>
      ${renderTextFields(job.payload || {})}
      ${Object.keys(job.payload || {}).length
        ? `<details class="detail-json-toggle" open><summary>完整 JSON</summary>${jsonBlock(job.payload, "payload 为空")}</details>`
        : '<div class="empty">payload 为空</div>'}
    </div>`,
    `<div class="detail-section">
      <h3 class="detail-h">结果 <span class="hint">result</span></h3>
      ${job.result == null
        ? `<div class="empty">${stateText === "done" ? "任务完成但没有结果" : `任务状态为 ${escapeHtml(stateText)}，暂无结果`}</div>`
        : `${jsonBlock(job.result, "result 为空")}${renderResultFiles(job)}`}
    </div>`,
    `<div class="detail-section">
      <h3 class="detail-h">错误 <span class="hint">error 原文</span></h3>
      ${errorText
        ? `<div class="detail-error-head"><span class="hint">等宽显示、保留换行</span>${copyButton(errorText, "复制错误")}</div>
           <pre class="log-box detail-error">${escapeHtml(errorText)}</pre>`
        : '<div class="empty">没有错误（任务正常结束）</div>'}
    </div>`,
    renderTimeline(job),
  ];
  box.innerHTML = body.join("");
  attachJobMedia(job);
}

function renderJobDetailError(message) {
  const box = $("#job-modal-body");
  if (!box) return;
  $("#job-modal-sub").textContent = "";
  $("#job-modal-meta").textContent = "读取失败";
  const cancelBtn = $("#job-modal-cancel");
  cancelBtn.disabled = true;
  cancelBtn.classList.add("hidden");
  box.innerHTML = `<div class="banner error">读取任务详情失败：${escapeHtml(message || "未知错误")}</div>
    <div class="empty">可以点「刷新详情」重试，或关闭弹窗后刷新任务列表。</div>`;
}

async function openJobDetail(jobId) {
  const modal = $("#modal-job");
  if (!modal) return;
  if (!state.jobDetailOpen) {
    state.jobDetailReturnFocus = document.activeElement;
  }
  state.jobDetailOpen = true;
  state.jobDetailId = jobId;
  state.jobDetailData = null;
  state.copyPayloads = {};
  modal.classList.remove("hidden");
  document.body.classList.add("modal-open");
  $("#job-modal-sub").textContent = jobId;
  $("#job-modal-meta").textContent = "";
  $("#job-modal-body").innerHTML = '<div class="empty">正在读取任务详情…</div>';
  const cancelBtn = $("#job-modal-cancel");
  cancelBtn.disabled = true;
  cancelBtn.classList.add("hidden");
  setTimeout(() => {
    const button = $("#job-modal-close");
    if (button) button.focus();
  }, 20);
  await loadJobDetail(jobId);
}

function closeJobDetail() {
  if (!state.jobDetailOpen) return;
  state.jobDetailOpen = false;
  state.jobDetailId = null;
  state.jobDetailData = null;
  state.copyPayloads = {};
  revokeJobBlobs();
  const modal = $("#modal-job");
  if (modal) modal.classList.add("hidden");
  document.body.classList.remove("modal-open");
  const box = $("#job-modal-body");
  if (box) box.innerHTML = "";
  const focus = state.jobDetailReturnFocus;
  state.jobDetailReturnFocus = null;
  if (focus && typeof focus.focus === "function") {
    try { focus.focus(); } catch (err) { /* 元素已不在 DOM 里 */ }
  }
}

async function refreshJobDetail() {
  if (!state.jobDetailId) return;
  await loadJobDetail(state.jobDetailId);
}

async function cancelJobFromDetail() {
  const jobId = state.jobDetailId;
  if (!jobId) return;
  const button = $("#job-modal-cancel");
  button.disabled = true;
  try {
    const data = await api(`/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
    toast(`任务 ${jobId} 已请求取消（状态 ${data.state}）`, "ok");
    await loadJobs().catch(() => {});
    if (state.jobDetailOpen && state.jobDetailId === jobId) await loadJobDetail(jobId);
  } catch (err) {
    toast(err.message || String(err), "error");
    button.disabled = false;
  }
}

async function cancelJob(jobId) {
  try {
    const data = await api(`/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
    toast(`任务 ${jobId} 已请求取消（状态 ${data.state}）`, "ok");
    await loadJobs();
  } catch (err) {
    toast(err.message || String(err), "error");
  }
}

/* ------------------------------------------------------------------ */
/* 系统                                                               */
/* ------------------------------------------------------------------ */
async function loadSystem() {
  const data = await api("/admin/system");
  state.system = data;
  renderSystem(data);
  renderAutostart(data.autostart || {});
  updateWarningBanner(data.listen || {});
}

function renderSystem(data) {
  const box = $("#system-grid");
  const service = data.service || {};
  const gpu = data.gpu || {};
  const disk = data.disk || {};
  const listen = data.listen || {};
  const jobs = data.jobs || {};
  const download = data.download || {};
  const loaded = Array.isArray(data.models_loaded) ? data.models_loaded : [];

  const diskRows = [
    ["模型", disk.models_mb],
    ["音色", disk.voices_mb],
    ["输出", disk.outputs_mb],
    ["日志", disk.logs_mb],
    ["数据", disk.data_mb],
    ["声纹", disk.speakers_mb],
  ].map(([label, value]) => `<div class="disk-row"><span class="muted">${escapeHtml(label)}</span><span>${escapeHtml(fmtMb(value))}</span></div>`).join("");

  box.innerHTML = `
    <div class="panel">
      <div class="panel-head"><h2>服务</h2><span class="pill ${data.auth_required ? "warn" : "muted"}">${data.auth_required ? "令牌已启用" : "未鉴权"}</span></div>
      <div class="kv"><span>版本</span><span>${escapeHtml(service.version || "--")}（API v${escapeHtml(service.api_version || "--")}）</span></div>
      <div class="kv"><span>Python</span><span>${escapeHtml(service.python || "--")}</span></div>
      <div class="kv"><span>平台</span><span>${escapeHtml(service.platform || "--")}</span></div>
      <div class="kv"><span>PID</span><span>${escapeHtml(service.pid || "--")}</span></div>
      <div class="kv"><span>启动时间</span><span>${escapeHtml(fmtClock(service.started_at))}</span></div>
      <div class="kv"><span>已运行</span><span>${escapeHtml(fmtDuration(service.uptime_seconds))}</span></div>
      <div class="kv"><span>监听</span><span>${escapeHtml(`${listen.host || "--"}:${listen.port || "--"}`)}</span></div>
      <div class="kv"><span>已加载模型</span><span>${escapeHtml(loaded.length ? loaded.join("、") : "无（0 模型）")}</span></div>
      <div class="kv"><span>yt-dlp</span><span><span id="ytdlp-version">${escapeHtml(download.yt_dlp_version || "未安装")}</span> <button id="ytdlp-check" class="small ghost" type="button">检查更新</button></span></div>
    </div>
    <div class="panel">
      <div class="panel-head"><h2>GPU</h2><span class="pill ${gpu.total_mb ? "ok" : "muted"}">${gpu.total_mb ? "正常" : "不可用"}</span></div>
      <div class="kv"><span>名称</span><span>${escapeHtml(gpu.name || "未检测到")}</span></div>
      <div class="kv"><span>驱动</span><span>${escapeHtml(gpu.driver || "--")}</span></div>
      <div class="kv"><span>显存</span><span>${escapeHtml(fmtMb(gpu.used_mb))} / ${escapeHtml(fmtMb(gpu.total_mb))}</span></div>
      <div class="kv"><span>空闲</span><span>${escapeHtml(fmtMb(gpu.free_mb))}</span></div>
      <div class="kv"><span>任务队列</span><span>排队 ${escapeHtml(jobs.queued || 0)} · 运行 ${escapeHtml(jobs.running || 0)}</span></div>
    </div>
    <div class="panel">
      <div class="panel-head"><h2>磁盘占用</h2><span class="pill muted">合计 ${escapeHtml(fmtMb(disk.total_mb))}</span></div>
      ${diskRows}
    </div>
    <div class="panel">
      <div class="panel-head"><h2>路径</h2></div>
      <div class="kv"><span>HOME</span><span>${escapeHtml((data.paths || {}).home || "--")}</span></div>
      <div class="kv"><span>模型</span><span>${escapeHtml((data.paths || {}).models || "--")}</span></div>
      <div class="kv"><span>配置</span><span>${escapeHtml((data.paths || {}).config_file || "--")}</span></div>
      <div class="kv"><span>日志</span><span>${escapeHtml((data.paths || {}).logs || "--")}</span></div>
    </div>`;

  const checkBtn = $("#ytdlp-check");
  if (checkBtn) {
    checkBtn.addEventListener("click", async () => {
      checkBtn.disabled = true;
      try {
        const status = await api("/media/download/status");
        const version = (status.yt_dlp || {}).version || "未安装";
        $("#ytdlp-version").textContent = version;
        toast(`当前 yt-dlp ${version}；升级：engine\\.venv\\Scripts\\python -m pip install -U yt-dlp`, "ok", 7000);
      } catch (err) {
        toast(err.message || String(err), "error");
      } finally {
        checkBtn.disabled = false;
      }
    });
  }
}

function renderAutostart(info) {
  const toggle = $("#autostart-toggle");
  const statePillEl = $("#autostart-state");
  const detail = $("#autostart-detail");
  toggle.checked = Boolean(info.enabled);
  toggle.disabled = info.supported === false;
  if (info.supported === false) {
    statePillEl.textContent = "非 Windows";
    statePillEl.className = "pill muted";
    detail.textContent = info.detail || "";
    return;
  }
  if (!info.exists) {
    statePillEl.textContent = "任务不存在";
    statePillEl.className = "pill warn";
    detail.textContent = "打开开关会创建登录时触发的计划任务。";
    return;
  }
  statePillEl.textContent = info.enabled ? "已启用" : "已禁用";
  statePillEl.className = `pill ${info.enabled ? "ok" : "warn"}`;
  detail.textContent = [
    `状态：${info.state || "--"}`,
    info.last_run_time ? `上次运行：${info.last_run_time}` : "",
    info.arguments ? `动作：${info.arguments}` : "",
  ].filter(Boolean).join("　|　");
}

async function toggleAutostart(enabled) {
  const toggle = $("#autostart-toggle");
  toggle.disabled = true;
  try {
    const data = await api("/admin/autostart", {
      method: "POST",
      body: JSON.stringify({ enabled }),
    });
    toast(enabled ? "开机自启已启用" : "开机自启已禁用", "ok");
    renderAutostart(data || {});
    if (state.view === "system") await loadSystem();
  } catch (err) {
    toast(err.message || String(err), "error", 7000);
    toggle.checked = !enabled;
  } finally {
    toggle.disabled = false;
  }
}

async function loadLogs() {
  const file = $("#log-file").value;
  const lines = Math.max(10, Math.min(5000, Number($("#log-lines").value) || 200));
  const output = $("#logs-output");
  output.textContent = "正在读取…";
  try {
    const data = await api(`/admin/logs?file=${encodeURIComponent(file)}&lines=${encodeURIComponent(lines)}`);
    $("#log-meta").textContent = data.exists === false
      ? (data.detail || "文件不存在")
      : `${data.filename || file} · ${data.encoding || "?"} · ${fmtBytes(data.size_bytes)} · 显示最后 ${(data.lines || []).length} 行`;
    output.textContent = (data.text || "").trim() || "（日志为空）";
    output.scrollTop = output.scrollHeight;
  } catch (err) {
    output.textContent = `读取失败：${err.message || String(err)}`;
    toast(err.message || String(err), "error");
  }
}

function updateWarningBanner(listen) {
  const box = $("#warning-banner");
  if (listen && listen.warning) {
    box.textContent = `⚠️ ${listen.warning}`;
    box.classList.remove("hidden");
  } else {
    box.classList.add("hidden");
  }
}

function updateHostWarning(host) {
  const value = String(host || "").trim();
  const loopback =
    value === "127.0.0.1" ||
    value === "localhost" ||
    value === "::1" ||
    value.startsWith("127.");
  if (loopback) {
    updateWarningBanner({});
    return;
  }
  updateWarningBanner({
    warning: `服务监听在非回环地址 ${value || "(未知)"}，局域网内其它机器可能访问管理接口；请务必设置 LOCALAI_TOKEN。`,
  });
}

/* ------------------------------------------------------------------ */
/* 重启                                                               */
/* ------------------------------------------------------------------ */
function confirmRestart() {
  showConfirm({
    title: "重启本地 AI 服务",
    message: "将先结束监听端口的进程，再在后台执行 start_ai_service.ps1。重启期间页面会短暂连不上。",
    requireText: "RESTART",
    onOk: async () => {
      try {
        const data = await api("/admin/restart", {
          method: "POST",
          body: JSON.stringify({ confirm: true }),
        });
        toast(data.detail || "服务将在约 2 秒后重启", "ok", 7000);
        waitForRestart();
      } catch (err) {
        toast(err.message || String(err), "error", 7000);
      }
    },
  });
}

async function waitForRestart() {
  if (state.restartPolling) return;
  state.restartPolling = true;
  setConnection(false, "等待重启…");
  let down = 0;
  const timer = setInterval(async () => {
    try {
      const response = await fetch("/health", { cache: "no-store" });
      if (response.ok) {
        clearInterval(timer);
        state.restartPolling = false;
        setConnection(true, "服务已恢复");
        toast("服务已重启完成", "ok");
        await refreshOverview(true);
        if (state.view === "system") await loadSystem();
      }
    } catch (err) {
      down += 1;
      if (down > 45) {
        clearInterval(timer);
        state.restartPolling = false;
        setConnection(false, "重启超时");
        toast("重启后 90 秒内仍未连上，请查看 logs/service.err.log", "error", 9000);
      }
    }
  }, 2000);
}

/* ------------------------------------------------------------------ */
/* 令牌 / 确认弹窗                                                     */
/* ------------------------------------------------------------------ */
function showTokenModal() {
  const modal = $("#modal-token");
  modal.classList.remove("hidden");
  const input = $("#token-input");
  input.value = state.token || "";
  setTimeout(() => input.focus(), 30);
}
function hideTokenModal() {
  $("#modal-token").classList.add("hidden");
}
function saveToken() {
  const value = $("#token-input").value.trim();
  if (!value) {
    toast("令牌不能为空；如果要清除请点「清除」", "error");
    return;
  }
  state.token = value;
  localStorage.setItem("localai_token", value);
  hideTokenModal();
  toast("令牌已保存到本浏览器", "ok");
  loadView(state.view, false);
}

let confirmHandler = null;
function showConfirm(options) {
  $("#confirm-title").textContent = options.title || "确认";
  $("#confirm-message").textContent = options.message || "";
  const wrap = $("#confirm-type-wrap");
  const input = $("#confirm-type");
  if (options.requireText) {
    wrap.classList.remove("hidden");
    input.value = "";
    wrap.dataset.requireText = options.requireText;
    setTimeout(() => input.focus(), 30);
  } else {
    wrap.classList.add("hidden");
    wrap.dataset.requireText = "";
  }
  confirmHandler = options.onOk || null;
  $("#modal-confirm").classList.remove("hidden");
}
function hideConfirm() {
  $("#modal-confirm").classList.add("hidden");
  confirmHandler = null;
}

/* ------------------------------------------------------------------ */
/* 初始化                                                             */
/* ------------------------------------------------------------------ */
function bindEvents() {
  $$(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => switchView(btn.dataset.view));
  });
  $("#refresh-btn").addEventListener("click", () => loadView(state.view, false));
  $("#auto-refresh").addEventListener("change", (event) => {
    state.autoRefresh = event.target.checked;
    toast(state.autoRefresh ? "自动刷新已开启" : "自动刷新已暂停", "ok", 1800);
  });

  $("#models-refresh").addEventListener("click", () => loadModels().catch((err) => toast(err.message, "error")));
  $("#settings-reload").addEventListener("click", () => loadSettings().catch((err) => toast(err.message, "error")));
  $("#settings-save").addEventListener("click", saveSettings);
  $("#jobs-refresh").addEventListener("click", () => loadJobs().catch((err) => toast(err.message, "error")));
  $("#jobs-filter").addEventListener("change", () => loadJobs().catch((err) => toast(err.message, "error")));
  // 双击任务行时不要让 3 秒自动刷新在两次点击之间重绘表格（会吃掉第二次点击）。
  $("#jobs-tbody").addEventListener("pointerdown", (event) => {
    if (event.target.closest("tr[data-job-id]")) state.lastJobInteraction = Date.now();
  }, true);
  $("#log-refresh").addEventListener("click", loadLogs);
  $("#log-file").addEventListener("change", loadLogs);
  $("#autostart-toggle").addEventListener("change", (event) => toggleAutostart(event.target.checked));
  $("#restart-btn").addEventListener("click", confirmRestart);

  $("#token-btn").addEventListener("click", showTokenModal);
  $("#token-save").addEventListener("click", saveToken);
  $("#token-cancel").addEventListener("click", hideTokenModal);
  $("#token-clear").addEventListener("click", () => {
    state.token = "";
    localStorage.removeItem("localai_token");
    $("#token-input").value = "";
    hideTokenModal();
    toast("已清除本浏览器保存的令牌", "ok");
  });
  $("#token-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter") saveToken();
  });

  $("#confirm-cancel").addEventListener("click", hideConfirm);
  $("#confirm-ok").addEventListener("click", () => {
    const wrap = $("#confirm-type-wrap");
    const required = wrap.dataset.requireText || "";
    if (required && $("#confirm-type").value.trim() !== required) {
      toast(`请输入 ${required} 以确认`, "error");
      return;
    }
    const handler = confirmHandler;
    hideConfirm();
    if (handler) handler();
  });

  // 任务详情弹窗
  $("#job-modal-close").addEventListener("click", closeJobDetail);
  $("#job-modal-close2").addEventListener("click", closeJobDetail);
  $("#job-modal-refresh").addEventListener("click", () => {
    refreshJobDetail().catch((err) => toast(err.message || String(err), "error"));
  });
  $("#job-modal-cancel").addEventListener("click", cancelJobFromDetail);
  $("#job-modal-body").addEventListener("click", (event) => {
    const button = event.target.closest("[data-copy-key]");
    if (!button) return;
    const text = state.copyPayloads[button.dataset.copyKey];
    if (text != null) copyText(text);
  });
  document.addEventListener("keydown", (event) => {
    if (!state.jobDetailOpen) return;
    if (event.key === "Escape") {
      event.preventDefault();
      closeJobDetail();
      return;
    }
    if (event.key !== "Tab") return;
    // 简单焦点圈：Tab 只在弹窗内循环，避免焦点跑到遮罩后面的列表。
    const focusable = $$('#modal-job button, #modal-job [href], #modal-job input, #modal-job select, #modal-job textarea, #modal-job [tabindex]:not([tabindex="-1"])')
      .filter((el) => !el.disabled && !el.classList.contains("hidden") && el.offsetParent !== null);
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  });

  $$(".modal").forEach((modal) => {
    modal.addEventListener("click", (event) => {
      if (event.target === modal) {
        if (modal.id === "modal-token") hideTokenModal();
        if (modal.id === "modal-confirm") hideConfirm();
        if (modal.id === "modal-job") closeJobDetail();
      }
    });
  });

  setInterval(() => {
    if (!state.autoRefresh || document.hidden) return;
    if (state.jobDetailOpen) return; // 详情弹窗打开时暂停列表刷新，避免打断阅读。
    if (state.view === "overview") refreshOverview(true).catch(() => {});
    // 刚点过任务行（可能是双击的第一下）时跳过这一次，保护双击手势。
    if (state.view === "jobs" && Date.now() - state.lastJobInteraction > 1500) {
      loadJobs().catch(() => {});
    }
  }, 3000);

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) return;
    if (state.view === "overview") refreshOverview(true).catch(() => {});
    if (state.view === "jobs" && !state.jobDetailOpen) loadJobs().catch(() => {});
  });
}

async function bootstrap() {
  bindEvents();
  setConnection(false, "连接中…");
  try {
    const caps = await api("/capabilities");
    state.authRequired = Boolean(caps.auth && caps.auth.required);
    updateHostWarning(caps.listen && caps.listen.host);
    if (state.authRequired && !state.token) {
      showTokenModal();
      return;
    }
  } catch (err) {
    // /capabilities 免鉴权；失败通常是服务没起来。
    showError(`无法连接服务：${err.message || String(err)}`);
  }
  await loadView("overview", true);
}

document.addEventListener("DOMContentLoaded", bootstrap);
