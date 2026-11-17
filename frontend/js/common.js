// Utilidades compartidas por las dos pantallas. Todo texto que viene del backend (en especial
// description_nl, que escribe un modelo) se inserta con textContent, nunca como HTML.

export const INCIDENT_LABELS = {
  fire: "Incendio",
  smoke: "Humo",
  accident: "Choque",
  fight: "Pelea",
  posible_altercado: "Posible altercado",
  posible_choque: "Posible choque",
};

export const CAMERA_HEALTH_LABELS = {
  corte_de_señal: "Sin señal",
  tapada: "Tapada",
  sin_analisis: "Sin análisis",
};

export const SELECTABLE_CATEGORIES = ["posible_altercado", "posible_choque", "fight", "accident", "fire"];

export const reasonLabel = (reason) =>
  INCIDENT_LABELS[reason] ?? CAMERA_HEALTH_LABELS[reason] ?? reason ?? "Evento";

export function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key in node) node[key] = value;
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

const API_KEY_STORAGE = "mu.apiKey";

const getApiKey = () => sessionStorage.getItem(API_KEY_STORAGE) ?? "";
const clearApiKey = () => sessionStorage.removeItem(API_KEY_STORAGE);

function authHeaders(extra = {}) {
  const key = getApiKey();
  return key ? { ...extra, "X-API-Key": key } : extra;
}

function rememberOperator(info) {
  currentOperator = { name: info.operator ?? "", cameras: info.cameras ?? null };
}

function backToLogin() {
  clearApiKey();
  location.reload();
}

export async function bootAuth() {
  try {
    const response = await fetch("/auth/check", { headers: authHeaders() });
    if (response.ok) rememberOperator(await response.json());
    if (response.status !== 401) return;
  } catch {
    return;
  }
  clearApiKey();

  await new Promise((resolve) => {
    const input = el("input", {
      type: "password",
      required: true,
      autocomplete: "current-password",
      "aria-label": "Clave de acceso",
    });
    const message = el("p", { class: "error", role: "alert" });
    const dialog = el(
      "dialog",
      { class: "dialog login" },
      el(
        "form",
        {
          onsubmit: async (event) => {
            event.preventDefault();
            sessionStorage.setItem(API_KEY_STORAGE, input.value);
            try {
              const check = await fetch("/auth/check", { headers: authHeaders() });
              if (check.ok) {
                rememberOperator(await check.json());
                dialog.close();
                dialog.remove();
                resolve();
                return;
              }
              message.textContent = "Clave incorrecta.";
            } catch {
              message.textContent = "No se pudo conectar con el servidor.";
            }
            clearApiKey();
            input.select();
          },
        },
        el("strong", { class: "brand" }, el("img", { src: "img/kenbun.svg", alt: "", class: "brand-logo" }), "KENBUN"),
        el("p", { class: "hint" }, "Ingresá tu clave de acceso personal."),
        input,
        message,
        el("button", { type: "submit", class: "primary" }, "Ingresar"),
      ),
    );
    dialog.addEventListener("cancel", (event) => event.preventDefault());
    document.body.append(dialog);
    dialog.showModal();
    input.focus();
  });
}

export async function api(path, { method = "GET", body } = {}) {
  const response = await fetch(path, {
    method,
    headers: authHeaders(body ? { "Content-Type": "application/json" } : {}),
    body: body ? JSON.stringify(body) : undefined,
  });
  if (response.status === 401) backToLogin();
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      if (typeof payload.detail === "string") detail = payload.detail;
    } catch {
      // el cuerpo no era JSON; alcanza con el statusText
    }
    throw new Error(`${response.status} — ${detail}`);
  }
  return response.json();
}

export async function resolveClipUrl(clipPath) {
  const { url } = await api(`/clip-url/${clipPath}`);
  return url;
}

// El backend guarda las marcas de tiempo en UTC sin zona. Se leen como UTC y se muestran en hora local.
export function parseTs(value) {
  if (!value) return null;
  const hasZone = /(Z|[+-]\d\d:?\d\d)$/.test(value);
  return new Date(hasZone ? value : `${value}Z`);
}

export function fmtDateTime(value) {
  const date = parseTs(value);
  return date ? date.toLocaleString("es-AR", { dateStyle: "short", timeStyle: "medium" }) : "—";
}

export function fmtTime(value) {
  const date = parseTs(value);
  return date ? date.toLocaleTimeString("es-AR", { timeStyle: "medium" }) : "—";
}

// <input type="datetime-local"> da hora local; la API espera UTC sin zona.
export function localInputToUtc(value) {
  return value ? new Date(value).toISOString().replace("Z", "") : null;
}

let currentOperator = { name: "", cameras: null };

export const getOperator = () => currentOperator;

export const scopedKey = (base) => `${base}.${currentOperator.name || "anonimo"}`;

// El navegador no reproduce RTSP. MediaMTX republica cada cámara también como HLS en el puerto 8888,
// bajo el mismo path que la fuente RTSP del catálogo.

export function hlsUrl(source) {
  const path = new URL(source).pathname.replace(/^\/+/, "");
  return `${location.protocol}//${location.hostname}:8888/${path}/index.m3u8`;
}

function streamAuthConfig() {
  const key = getApiKey();
  if (!key) return {};
  const authorization = `Basic ${btoa(`operador:${key}`)}`;
  return {
    xhrSetup: (xhr) => xhr.setRequestHeader("Authorization", authorization),
    fetchSetup: (context, init) => {
      const headers = new Headers(init.headers);
      headers.set("Authorization", authorization);
      return new Request(context.url, { ...init, headers });
    },
  };
}

const STALL_SECONDS = 6;
const RETRY_MS = 3000;

// Reproduce un stream HLS y lo reconecta solo si cae o se congela. onState recibe "up" o "down".
export function playLive(video, url, onState) {
  let hls = null;
  let retryTimer = null;
  let state = null;
  let lastTime = -1;
  let lastProgress = Date.now();
  let stopped = false;

  const setState = (next) => {
    if (next !== state) {
      state = next;
      onState(next);
    }
  };

  const teardown = () => {
    if (hls) {
      hls.destroy();
      hls = null;
    }
    video.removeAttribute("src");
    video.load();
  };

  const scheduleRetry = () => {
    setState("down");
    teardown();
    clearTimeout(retryTimer);
    if (!stopped) retryTimer = setTimeout(start, RETRY_MS);
  };

  function start() {
    if (stopped) return;
    lastTime = -1;
    lastProgress = Date.now();
    if (window.Hls?.isSupported()) {
      hls = new window.Hls({ lowLatencyMode: true, ...streamAuthConfig() });
      hls.on(window.Hls.Events.ERROR, (_event, data) => {
        if (data.fatal) scheduleRetry();
      });
      hls.loadSource(url);
      hls.attachMedia(video);
    } else if (video.canPlayType("application/vnd.apple.mpegurl")) {
      video.src = url;
      video.addEventListener("error", scheduleRetry, { once: true });
    } else {
      setState("down");
      return;
    }
    video.play().catch(() => {});
  }

  video.addEventListener("playing", () => setState("up"));

  const watchdog = setInterval(() => {
    if (stopped || state !== "up") return;
    if (video.currentTime !== lastTime) {
      lastTime = video.currentTime;
      lastProgress = Date.now();
    } else if (Date.now() - lastProgress > STALL_SECONDS * 1000) {
      scheduleRetry();
    }
  }, 2000);

  start();

  return () => {
    stopped = true;
    clearTimeout(retryTimer);
    clearInterval(watchdog);
    teardown();
  };
}

export const SELECTED_CAMERAS_KEY = "mu.cameras";

function selectedChannelCameras() {
  try {
    const raw = localStorage.getItem(scopedKey(SELECTED_CAMERAS_KEY));
    return raw === null ? null : JSON.parse(raw);
  } catch {
    return null;
  }
}

function createChannel(path) {
  const listeners = new Set();
  let socket = null;

  function connect() {
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const cameras = selectedChannelCameras();
    const query = cameras === null ? "" : `?cameras=${cameras.map(encodeURIComponent).join(",")}`;
    const current = new WebSocket(`${protocol}//${location.host}${path}${query}`);
    socket = current;
    current.addEventListener("open", () => {
      const key = getApiKey();
      if (key) current.send(JSON.stringify({ api_key: key }));
    });
    current.addEventListener("message", (message) => {
      let payload;
      try {
        payload = JSON.parse(message.data);
      } catch {
        return;
      }
      listeners.forEach((listener) => listener(payload));
    });
    current.addEventListener("close", (event) => {
      if (event.code === 4401) return backToLogin();
      if (socket !== current) return;
      socket = null;
      setTimeout(connect, RETRY_MS);
    });
  }

  return {
    subscribe(listener) {
      listeners.add(listener);
      if (!socket) connect();
    },
    reconnect() {
      if (!socket) return;
      const previous = socket;
      socket = null;
      previous.close();
      connect();
    },
  };
}

const alertChannel = createChannel("/alerts");
const healthChannel = createChannel("/camera-status");

export const onAlert = (listener) => alertChannel.subscribe(listener);
export const onCameraHealth = (listener) => healthChannel.subscribe(listener);

export function reconnectAlerts() {
  alertChannel.reconnect();
  healthChannel.reconnect();
}

const PENDING_REFRESH_MS = 15000;

export function mountTopbar(container, active) {
  const cameras = new Map();

  const parkStatus = el("button", { type: "button", class: "park-status", "aria-haspopup": "dialog" }, "Status cámaras: …");
  const faultList = el("div", { class: "fault-list" });
  const faultDialog = el(
    "dialog",
    { class: "dialog" },
    el("form", { method: "dialog", class: "dialog-head" }, el("strong", {}, "Cámaras con falla"), el("button", {}, "Cerrar")),
    faultList,
  );
  document.body.append(faultDialog);
  const pendingLink = el("span", { class: "nav-badge pending-counter", hidden: true, title: "Eventos sin confirmar" });
  const alertBadge = el("span", { class: "nav-badge alert-counter", hidden: true, title: "Alertas en el tablero" });
  const operatorName = el("span", { class: "operator-name" }, `Operador: ${currentOperator.name || "—"}`);

  function setAlertCount(count) {
    alertBadge.textContent = String(count);
    alertBadge.title = `Alertas en el tablero: ${count}`;
    alertBadge.hidden = count === 0;
  }

  function storedAlertCount() {
    try {
      const raw = localStorage.getItem(scopedKey("mu.alerts"));
      const stored = raw === null ? [] : JSON.parse(raw);
      const filter = selectedChannelCameras();
      return stored.filter((alert) => filter === null || filter.includes(alert.camera_id)).length;
    } catch {
      return 0;
    }
  }

  container.append(
    el("strong", { class: "brand" }, el("img", { src: "img/kenbun.svg", alt: "", class: "brand-logo" }), "KENBUN"),
    el(
      "nav",
      {},
      el("a", { href: "#tablero", class: "nav-button", "data-view": "tablero" }, "Tablero", alertBadge),
      el("a", { href: "#revision", class: "nav-button", "data-view": "revision" }, "Revisión", pendingLink),
    ),
    el("span", { class: "spacer" }),
    parkStatus,
    operatorName,
  );

  const faultyCameras = () => [...cameras.values()].filter((health) => health.status !== "ok");

  function renderFaults() {
    const faulty = faultyCameras();
    faultList.replaceChildren(
      ...(faulty.length === 0
        ? [el("p", { class: "empty" }, "Todas las cámaras funcionan correctamente.")]
        : faulty.map((health) =>
            el(
              "div",
              { class: "fault-row" },
              el("strong", {}, health.camera_label),
              el("span", { class: "fault-kind" }, reasonLabel(health.status)),
              el("small", {}, `${health.camera_id} · desde ${fmtTime(health.since)}`),
            ),
          )),
    );
  }

  function renderPark() {
    const faulty = faultyCameras();
    if (cameras.size === 0) {
      parkStatus.className = "park-status";
      parkStatus.textContent = "Status cámaras: …";
    } else if (faulty.length === 0) {
      parkStatus.className = "park-status ok";
      parkStatus.textContent = "Status cámaras: OK";
    } else {
      parkStatus.className = "park-status bad";
      parkStatus.textContent = `Status cámaras: ${faulty.length} con falla`;
    }
    if (faultDialog.open) renderFaults();
  }

  parkStatus.addEventListener("click", () => {
    renderFaults();
    faultDialog.showModal();
  });

  async function refreshPending() {
    try {
      const pending = await api("/events/pending-review");
      pendingLink.textContent = String(pending.length);
      pendingLink.title = `Sin confirmar: ${pending.length}`;
      pendingLink.hidden = pending.length === 0;
    } catch {
      pendingLink.textContent = "?";
      pendingLink.title = "No se pudo consultar los eventos sin confirmar";
      pendingLink.hidden = false;
    }
  }

  onCameraHealth((health) => {
    cameras.set(health.camera_id, health);
    renderPark();
  });

  renderPark();

  refreshPending();
  setInterval(refreshPending, PENDING_REFRESH_MS);

  setAlertCount(storedAlertCount());

  function setActive(name) {
    for (const link of container.querySelectorAll("nav a[data-view]")) {
      if (link.dataset.view === name) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    }
  }

  return { refreshPending, setAlertCount, setActive };
}
