import {
  api,
  el,
  fmtDateTime,
  fmtTime,
  hlsUrl,
  localInputToUtc,
  onAlert,
  onCameraHealth,
  parseTs,
  playLive,
  reasonLabel,
  reconnectAlerts,
  resolveClipUrl,
  SELECTED_CAMERAS_KEY,
  scopedKey,
} from "./common.js?v=16";
import { vehiclesLabel } from "./vehicles.js?v=16";
import { createReviewPanel } from "./review-panel.js?v=16";
import { registerView, topbar } from "./shell.js?v=16";

const store = {
  read(key, fallback) {
    try {
      const raw = localStorage.getItem(scopedKey(key));
      return raw === null ? fallback : JSON.parse(raw);
    } catch {
      return fallback;
    }
  },
  write(key, value) {
    try {
      localStorage.setItem(scopedKey(key), JSON.stringify(value));
    } catch {
    }
  },
};

let catalog = [];

const gridEl = document.getElementById("camera-grid");
const layoutSelect = document.getElementById("layout-select");
const tiles = new Map();
const healthByCamera = new Map();

layoutSelect.value = String(store.read("mu.cols", 3));
layoutSelect.addEventListener("change", () => {
  store.write("mu.cols", Number(layoutSelect.value));
  applyLayout();
});

function applyLayout() {
  gridEl.style.setProperty("--cols", layoutSelect.value);
}

function selectedCameras() {
  return store.read(SELECTED_CAMERAS_KEY, null) ?? catalog.slice(0, 6).map((camera) => camera.camera_id);
}

function renderGrid() {
  tiles.forEach((tile) => tile.stop());
  tiles.clear();
  gridEl.replaceChildren();

  const selected = selectedCameras();
  const cameras = catalog.filter((camera) => selected.includes(camera.camera_id));

  if (cameras.length === 0) {
    gridEl.append(el("p", { class: "empty" }, "No hay cámaras en el tablero. Usá «Elegir cámaras»."));
    return;
  }

  for (const camera of cameras) {
    const video = el("video", { muted: true, autoplay: true, playsinline: true });
    const overlay = el("div", { class: "tile-overlay" }, "SIN SEÑAL");
    const dot = el("span", { class: "dot down" });
    const badge = el("span", { class: "tile-health" });
    const tile = el(
      "figure",
      { class: "tile down" },
      video,
      overlay,
      el("figcaption", {}, dot, camera.label, badge),
    );
    gridEl.append(tile);

    const stop = playLive(video, hlsUrl(camera.source), (state) => {
      tile.classList.toggle("down", state === "down");
      dot.classList.toggle("down", state === "down");
    });
    tiles.set(camera.camera_id, { stop, tile, badge });
    applyHealth(camera.camera_id);
  }
}

function applyHealth(cameraId) {
  const entry = tiles.get(cameraId);
  if (!entry) return;
  const status = healthByCamera.get(cameraId) ?? "ok";
  entry.tile.classList.toggle("fault", status !== "ok");
  entry.badge.textContent = status === "ok" ? "" : reasonLabel(status);
}

onCameraHealth((health) => {
  healthByCamera.set(health.camera_id, health.status);
  applyHealth(health.camera_id);
});

function openCameraPicker() {
  const dialog = document.getElementById("pick-dialog");
  const list = document.getElementById("pick-list");
  const selected = new Set(selectedCameras());
  list.replaceChildren(
    ...catalog.map((camera) =>
      el(
        "label",
        { class: "check" },
        el("input", {
          type: "checkbox",
          checked: selected.has(camera.camera_id),
          onchange: (event) => {
            if (event.target.checked) selected.add(camera.camera_id);
            else selected.delete(camera.camera_id);
            store.write(SELECTED_CAMERAS_KEY, [...selected]);
            renderGrid();
            renderAlerts();
            reconnectAlerts(); // el servidor deja de mandar las alertas de las cámaras destildadas
          },
        }),
        `${camera.label} `,
        el("small", {}, camera.camera_id),
      ),
    ),
  );
  if (catalog.length === 0) list.append(el("p", { class: "empty" }, "El catálogo de cámaras está vacío."));
  dialog.showModal();
}

document.getElementById("pick-cameras").addEventListener("click", openCameraPicker);

const MAX_ALERTS = 200;
const DESCRIPTION_POLL_MS = 3000;
const DESCRIPTION_GIVE_UP_MS = 10 * 60 * 1000;

const alertListEl = document.getElementById("alert-list");
const soundToggle = document.getElementById("sound-toggle");
let alerts = store.read("mu.alerts", []).map((alert) => ({ ...alert, fresh: false }));
let soundOn = store.read("mu.sound", false);
let audioContext = null;

function persistAlerts() {
  store.write("mu.alerts", alerts.slice(0, MAX_ALERTS));
}

function alertAllowed(alert) {
  const filter = store.read(SELECTED_CAMERAS_KEY, null);
  return filter === null || filter.includes(alert.camera_id);
}

const BEEP_COUNT = 4;
const BEEP_SECONDS = 0.18;
const BEEP_GAP_SECONDS = 0.14;

function beep() {
  if (!soundOn || !audioContext) return;
  const start = audioContext.currentTime;
  for (let i = 0; i < BEEP_COUNT; i += 1) {
    const at = start + i * (BEEP_SECONDS + BEEP_GAP_SECONDS);
    const oscillator = audioContext.createOscillator();
    const gain = audioContext.createGain();
    oscillator.type = "square";
    oscillator.frequency.value = 880;
    gain.gain.value = 0.15;
    oscillator.connect(gain).connect(audioContext.destination);
    oscillator.start(at);
    oscillator.stop(at + BEEP_SECONDS);
  }
}

function syncSoundToggle() {
  soundToggle.setAttribute("aria-pressed", String(soundOn));
  soundToggle.textContent = `Sonido: ${soundOn ? "encendido" : "apagado"}`;
}

function ensureAudio() {
  audioContext ??= new AudioContext();
  if (audioContext.state === "suspended") audioContext.resume();
}

soundToggle.addEventListener("click", () => {
  soundOn = !soundOn;
  store.write("mu.sound", soundOn);
  if (soundOn) ensureAudio();
  syncSoundToggle();
  if (soundOn) beep();
});

// La preferencia se recuerda entre pantallas, pero el navegador solo deja sonar tras un gesto del usuario:
// si quedó encendida, el audio se habilita con el primer clic en la página.
syncSoundToggle();
if (soundOn) document.addEventListener("pointerdown", ensureAudio, { once: true });

function renderAlerts() {
  const visible = alerts.filter(alertAllowed);
  topbar.setAlertCount(visible.length);
  alertListEl.replaceChildren(
    ...(visible.length === 0
      ? [el("p", { class: "empty" }, "Sin alertas. Las que lleguen quedan acá hasta que las borres.")]
      : visible.map(alertRow)),
  );
}

function alertRow(alert) {
  const reviewed = alert.review_status && alert.review_status !== "pending";
  const description = alert.description_nl
    ? el("span", { class: "desc" }, alert.description_nl)
    : el("span", { class: "desc pending" }, "Generando descripción…");

  return el(
    "article",
    { class: `alert-row ${alert.fresh ? "fresh" : ""}`, dataset: { id: alert.event_id } },
    el("time", {}, fmtTime(alert.timestamp)),
    el("strong", { class: "kind" }, reasonLabel(alert.trigger_reason)),
    el("span", { class: "cam" }, alert.camera_label),
    el("span", { class: "desc-box" }, description, alert.vehicles?.length ? el("small", { class: "vehicles-note" }, `Vehículos: ${vehiclesLabel(alert.vehicles)}`) : null),
    el(
      "span",
      { class: "actions" },
      alert.description_edited ? el("span", { class: "badge edited", title: "La descripción fue corregida por un operador" }, "Corregida") : null,
      reviewed
        ? el("span", { class: `badge ${alert.review_status}` }, alert.review_status === "confirmed" ? "Confirmada" : "Descartada")
        : [
            el("button", { type: "button", class: "primary", title: "Abre el evento para confirmarlo o descartarlo", onclick: () => openReview(alert) }, "Revisar"),
            el("button", { type: "button", class: "ghost", title: "La saca de las alertas; sigue pendiente en Revisión", onclick: () => ignore(alert) }, "Ignorar"),
          ],
    ),
  );
}

const reviewPanel = createReviewPanel(document.getElementById("review-overlay"), {
  places: () => Object.fromEntries(catalog.filter((camera) => camera.ubicacion).map((camera) => [camera.camera_id, camera.ubicacion])),
  onResolved: (event) => {
    alerts = alerts.filter((item) => item.event_id !== event.event_id);
    persistAlerts();
    renderAlerts();
    topbar.refreshPending();
  },
});

async function openReview(alert) {
  try {
    const event = await api(`/events/${encodeURIComponent(alert.event_id)}`);
    if (event.review_status !== "pending") {
      alerts = alerts.filter((item) => item.event_id !== alert.event_id);
      persistAlerts();
      renderAlerts();
      return;
    }
    reviewPanel.open(event);
  } catch (error) {
    window.alert(`No se pudo abrir el evento: ${error.message}`);
  }
}

// Ignorar saca la alerta de esta lista, pero el evento sigue sin confirmar en la cola de Revisión.
function ignore(alert) {
  alerts = alerts.filter((item) => item.event_id !== alert.event_id);
  persistAlerts();
  renderAlerts();
}

const clipDialog = document.getElementById("clip-dialog");
const clipVideo = document.getElementById("clip-video");
clipDialog.addEventListener("close", () => {
  clipVideo.pause();
  clipVideo.removeAttribute("src");
  clipVideo.load();
});

function openClip(alert) {
  document.getElementById("clip-title").textContent = `${reasonLabel(alert.trigger_reason)} — ${alert.camera_label} — ${fmtDateTime(alert.timestamp)}`;
  clipDialog.showModal();
  resolveClipUrl(alert.clip_path)
    .then((url) => {
      clipVideo.src = url;
      return clipVideo.play();
    })
    .catch(() => {});
}

// El canal en tiempo real avisa apenas se inserta el evento, sin esperar al módulo de descripción:
// la fila nace sin texto y se completa preguntando por el evento hasta que aparece.
async function completeAlert(alert) {
  const deadline = (parseTs(alert.timestamp)?.getTime() ?? Date.now()) + DESCRIPTION_GIVE_UP_MS;
  while (Date.now() < deadline) {
    if (!alerts.includes(alert)) return;
    try {
      const event = await api(`/events/${alert.event_id}`);
      alert.clip_path = event.clip_path;
      alert.review_status = event.review_status;
      alert.description_edited = event.description_edited;
      alert.vehicles = event.vehicles;
      if (event.description_nl) {
        alert.description_nl = event.description_nl;
      }
      persistAlerts();
      renderAlerts();
      if (event.description_nl) return;
    } catch {
    }
    await new Promise((resolve) => setTimeout(resolve, DESCRIPTION_POLL_MS));
  }
}

onAlert((message) => {
  if (alerts.some((alert) => alert.event_id === message.event_id)) return;

  const alert = { ...message, review_status: "pending", fresh: true };
  alerts.unshift(alert);
  persistAlerts();
  renderAlerts();
  if (alertAllowed(alert)) beep();
  setTimeout(() => {
    alert.fresh = false;
    persistAlerts();
  }, 4000);
  completeAlert(alert);
});

document.getElementById("alert-clear").addEventListener("click", () => {
  // Solo vacía la lista del tablero: las alertas sin confirmar siguen en la cola de Revisión, que sale del servidor.
  if (alerts.length === 0) return;
  if (!window.confirm("Se quitan las alertas del tablero. Las que siguen sin confirmar quedan en Revisión y los eventos siguen en el sistema. ¿Continuar?")) return;
  alerts = [];
  persistAlerts();
  renderAlerts();
});

const chatLog = document.getElementById("chat-log");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatSend = document.getElementById("chat-send");
const chatCamera = document.getElementById("chat-camera");
const CHAT_SLOW_AFTER_SECONDS = 15;
const CHAT_STORE_KEY = "mu.chat";
const CHAT_MAX_STORED = 40;
const CHAT_MAX_SENT = 12;

let chat = loadChat();
let chatEpoch = 0;

function loadChat() {
  const saved = store.read(CHAT_STORE_KEY, null);
  return {
    messages: Array.isArray(saved?.messages) ? saved.messages : [],
    excluded: Array.isArray(saved?.excluded) ? saved.excluded : [],
  };
}

function saveChat() {
  chat.messages = chat.messages.slice(-CHAT_MAX_STORED);
  store.write(CHAT_STORE_KEY, chat);
}

function chatHistory() {
  return chat.messages
    .filter((message) => message.role === "user" || message.role === "bot")
    .slice(-CHAT_MAX_SENT)
    .map((message) =>
      message.role === "user"
        ? { role: "user", content: message.text }
        : { role: "assistant", content: message.text, event_ids: (message.events ?? []).map((event) => event.event_id) },
    );
}

function addMessage(kind, ...content) {
  const node = el("div", { class: `msg ${kind}` }, ...content);
  chatLog.append(node);
  chatLog.scrollTop = chatLog.scrollHeight;
  return node;
}

function citedEvent(event) {
  const holder = el("div", { class: "clip-slot" });
  const toggle = el("button", { type: "button" }, "▶ Ver clip");
  toggle.addEventListener("click", async () => {
    if (holder.firstChild) {
      holder.replaceChildren();
      toggle.textContent = "▶ Ver clip";
      return;
    }
    toggle.textContent = "■ Ocultar clip";
    try {
      const src = await resolveClipUrl(event.clip_path);
      holder.append(el("video", { controls: true, autoplay: true, playsinline: true, src }));
    } catch (error) {
      holder.append(el("p", { class: "error" }, `No se pudo cargar el clip: ${error.message}`));
    }
  });
  return el(
    "div",
    { class: "cited" },
    el(
      "div",
      { class: "cited-head" },
      el("strong", {}, reasonLabel(event.trigger_reason)),
      ` · ${event.camera_label} · ${fmtDateTime(event.start_ts)} `,
      event.description_edited ? el("span", { class: "badge edited", title: "La descripción fue corregida por un operador" }, "Corregida") : null,
    ),
    el("p", {}, event.description_nl ?? ""),
    event.vehicles?.length ? el("p", { class: "vehicles-note" }, `Vehículos: ${vehiclesLabel(event.vehicles)}`) : null,
    toggle,
    holder,
  );
}

function renderChat() {
  chatLog.replaceChildren();
  if (chat.messages.length === 0) {
    chatLog.append(el("p", { class: "empty" }, "Hacé una pregunta sobre lo que pasó. Después podés seguir la conversación."));
    return;
  }
  for (const message of chat.messages) {
    if (message.role === "user") addMessage("user", message.text);
    else if (message.role === "error") addMessage("error", message.text);
    else {
      addMessage(
        "bot",
        el("p", {}, message.text),
        message.searchQuery ? el("p", { class: "search-note" }, `Busqué: «${message.searchQuery}»`) : null,
        ...(message.events ?? []).map(citedEvent),
      );
    }
  }
}

function compactEvent(event) {
  return {
    event_id: event.event_id,
    trigger_reason: event.trigger_reason,
    camera_label: event.camera_label,
    start_ts: event.start_ts,
    description_nl: event.description_nl,
    vehicles: event.vehicles,
    clip_path: event.clip_path,
  };
}

document.getElementById("chat-clear").addEventListener("click", () => {
  const hasContent = chat.messages.length > 0 || chat.excluded.length > 0 || chatInput.value.trim() !== "";
  if (hasContent && !window.confirm("Se borra la conversación y el contexto guardados en este navegador. ¿Continuar?")) return;
  chatEpoch += 1;
  chat = { messages: [], excluded: [] };
  try {
    localStorage.removeItem(scopedKey(CHAT_STORE_KEY));
  } catch {
  }
  chatInput.value = "";
  chatCamera.value = "";
  document.getElementById("chat-start").value = "";
  document.getElementById("chat-end").value = "";
  chatSend.disabled = false;
  renderChat();
});

chatForm.addEventListener("submit", async (submit) => {
  submit.preventDefault();
  const query = chatInput.value.trim();
  if (!query) return;

  const body = { query, history: chatHistory(), excluded_event_ids: chat.excluded };
  if (chatCamera.value) body.camera_id = chatCamera.value;
  const start = localInputToUtc(document.getElementById("chat-start").value);
  const end = localInputToUtc(document.getElementById("chat-end").value);
  if (start) body.start_ts = start;
  if (end) body.end_ts = end;

  const epoch = chatEpoch;
  chat.messages.push({ role: "user", text: query });
  saveChat();
  renderChat();
  chatInput.value = "";
  chatSend.disabled = true;
  const status = el("span", {}, "Consultando…");
  const waiting = addMessage("bot", el("span", { class: "spinner", "aria-hidden": "true" }), status);
  waiting.setAttribute("role", "status");

  const startedAt = Date.now();
  const timer = setInterval(() => {
    const seconds = Math.floor((Date.now() - startedAt) / 1000);
    status.textContent =
      seconds < CHAT_SLOW_AFTER_SECONDS
        ? `Consultando… ${seconds} s`
        : `Consultando… ${seconds} s — está tardando más de lo normal; puede estar cargando el modelo o la GPU estar ocupada`;
  }, 1000);

  try {
    const response = await api("/chat", { method: "POST", body });
    if (epoch !== chatEpoch) return;
    chat.messages.push({
      role: "bot",
      text: response.answer,
      searchQuery: response.search_query,
      events: response.events.map(compactEvent),
    });
    chat.excluded = response.excluded_event_ids ?? [];
  } catch (error) {
    if (epoch !== chatEpoch) return;
    chat.messages.push({ role: "error", text: `No se pudo completar la consulta: ${error.message}` });
  } finally {
    clearInterval(timer);
    if (epoch === chatEpoch) {
      saveChat();
      renderChat();
      chatSend.disabled = false;
      chatLog.scrollTop = chatLog.scrollHeight;
    }
  }
});

chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

async function refreshReviewState() {
  for (const alert of alerts.filter((item) => item.description_nl && item.review_status === "pending")) {
    try {
      const event = await api(`/events/${alert.event_id}`);
      if (event.review_status !== "pending") alerts = alerts.filter((item) => item !== alert);
      else Object.assign(alert, { review_status: event.review_status, description_nl: event.description_nl, description_edited: event.description_edited, vehicles: event.vehicles });
    } catch {
    }
  }
  persistAlerts();
  renderAlerts();
}

async function init() {
  applyLayout();
  renderAlerts();
  renderChat();
  alerts.filter((alert) => !alert.description_nl).forEach(completeAlert);
  refreshReviewState();
  try {
    catalog = await api("/cameras");
  } catch (error) {
    gridEl.append(el("p", { class: "empty" }, `No se pudo cargar el catálogo de cámaras: ${error.message}`));
    return;
  }
  chatCamera.append(...catalog.map((camera) => el("option", { value: camera.camera_id }, camera.label)));
  if (store.read(SELECTED_CAMERAS_KEY, null) === null) {
    store.write(SELECTED_CAMERAS_KEY, selectedCameras());
    reconnectAlerts();
  }
  renderGrid();
}

init();

registerView("tablero", { el: document.querySelector("main.layout"), onShow: refreshReviewState });
