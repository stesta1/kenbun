import { SELECTABLE_CATEGORIES, api, el, fmtDateTime, reasonLabel, resolveClipUrl } from "./common.js?v=16";
import { VEHICLE_REASONS, createVehicleEditor } from "./vehicles.js?v=16";
import { goTo, registerView, shell, topbar } from "./shell.js?v=16";

const QUEUE_REFRESH_MS = 10000;

const queueList = document.getElementById("queue-list");
const queueCount = document.getElementById("queue-count");
const detailEmpty = document.getElementById("detail-empty");
const detail = document.getElementById("detail");
const decision = document.getElementById("decision");
const video = document.getElementById("review-video");
const meta = document.getElementById("meta");
const noAlertNote = document.getElementById("no-alert-note");
const description = document.getElementById("description");
const categorySelect = document.getElementById("category");
categorySelect.append(...SELECTABLE_CATEGORIES.map((reason) => el("option", { value: reason }, reasonLabel(reason))));
const vehicleEditor = createVehicleEditor();
document.getElementById("vehicles-mount").append(vehicleEditor.root);
const confirmButton = document.getElementById("confirm");
const statusMsg = document.getElementById("status-msg");
const actionButtons = [confirmButton, document.getElementById("dismiss")];

// Una alerta que llega desde el tablero («Revisar») se abre acá por su id, aunque no esté en la cola.
let focused = null;
let places = {}; // camera_id -> calles donde está la cámara
let queue = [];
let selected = null;

function say(message, isError = false) {
  statusMsg.textContent = message;
  statusMsg.className = isError ? "error" : "";
}

function visibleQueue() {
  return focused && !queue.some((event) => event.event_id === focused.event_id) ? [focused, ...queue] : queue;
}

function renderQueue() {
  const items = visibleQueue();
  queueCount.textContent = items.length === 0 ? "" : `${items.length}`;
  if (items.length === 0) {
    queueList.replaceChildren(el("li", { class: "empty" }, "No hay eventos sin confirmar."));
    return;
  }
  queueList.replaceChildren(
    ...items.map((event) =>
      el(
        "li",
        {},
        el(
          "button",
          {
            type: "button",
            class: `queue-item ${selected?.event_id === event.event_id ? "active" : ""}`,
            onclick: () => select(event),
          },
          el("strong", {}, reasonLabel(event.trigger_reason)),
          el("span", {}, fmtDateTime(event.start_ts)),
          el("small", {}, event.camera_label),
        ),
      ),
    ),
  );
}

function syncVehicles() {
  vehicleEditor.setVisible(VEHICLE_REASONS.includes(categorySelect.value));
}

function select(event) {
  selected = event;
  say("");
  renderQueue();

  detailEmpty.hidden = true;
  detail.hidden = false;
  decision.hidden = false;

  resolveClipUrl(event.clip_path)
    .then((url) => {
      if (selected !== event) return;
      video.src = url;
      return video.play();
    })
    .catch(() => {});

  const rows = [
    ["Motivo de disparo", reasonLabel(event.trigger_reason)],
    ["Cámara", `${event.camera_label} (${event.camera_id})`],
    ["Ubicación", places[event.camera_id] ?? "—"],
    ["Objetos seguidos", event.object_classes.join(", ") || "—"],
    ["Inicio", fmtDateTime(event.start_ts)],
    ["Fin", fmtDateTime(event.end_ts)],
    ["Confianza", event.confidence === null ? "—" : `${Math.round(event.confidence * 100)} %`],
  ];
  meta.replaceChildren(...rows.flatMap(([term, value]) => [el("dt", {}, term), el("dd", {}, value)]));

  noAlertNote.textContent = ["accident", "fire", "fight"].includes(event.trigger_reason)
    ? ""
    : "Este evento no generó una alerta: la regla que lo disparó reconoce un patrón de movimiento compatible, " +
      "sin que el detector haya clasificado el hecho. No es una falla del canal de alertas.";

  categorySelect.value = event.trigger_reason;
  description.value = event.description_nl ?? "";
  description.disabled = false;
  vehicleEditor.setVehicles(event.vehicles);
  syncVehicles();
  const waiting = !event.description_nl && event.status !== "failed";
  actionButtons.forEach((button) => (button.disabled = false));
  confirmButton.disabled = waiting;
  if (waiting) say("La descripción todavía se está generando.");
}

categorySelect.addEventListener("change", syncVehicles);
description.addEventListener("input", () => {
  if (selected && description.value.trim()) confirmButton.disabled = false;
});

function clearSelection() {
  selected = null;
  video.pause();
  video.removeAttribute("src");
  video.load();
  detail.hidden = true;
  decision.hidden = true;
  detailEmpty.hidden = false;
}

async function decide(action) {
  if (!selected) return;
  const current = selected;
  const text = description.value.trim();
  const needsVehicles = VEHICLE_REASONS.includes(categorySelect.value);
  if (action === "confirmed") {
    if (!text) return say("Escribí la descripción antes de confirmar.", true);
    if (needsVehicles && !vehicleEditor.isComplete()) return say("Un choque necesita al menos un vehículo, con su tipo.", true);
  }

  actionButtons.forEach((button) => (button.disabled = true));
  try {
    if (action === "confirmed") {
      await api(`/events/${current.event_id}/details`, {
        method: "POST",
        body: {
          description_nl: text,
          trigger_reason: categorySelect.value !== current.trigger_reason ? categorySelect.value : null,
          vehicles: needsVehicles ? vehicleEditor.getVehicles() : [],
        },
      });
    } else {
      await api(`/events/${current.event_id}/review`, { method: "POST", body: { review_status: "dismissed" } });
    }
    const index = visibleQueue().findIndex((event) => event.event_id === current.event_id);
    queue = queue.filter((event) => event.event_id !== current.event_id);
    if (focused?.event_id === current.event_id) focused = null;
    topbar.refreshPending();
    clearSelection();
    actionButtons.forEach((button) => (button.disabled = false));
    if (shell.returnTo === "tablero") {
      goTo("tablero");
      return;
    }
    const rest = visibleQueue();
    const next = rest[Math.min(index, rest.length - 1)];
    if (next) select(next);
    else renderQueue();
  } catch (error) {
    say(`No se pudo guardar: ${error.message}`, true);
    actionButtons.forEach((button) => (button.disabled = false));
  }
}

confirmButton.addEventListener("click", () => decide("confirmed"));
document.getElementById("dismiss").addEventListener("click", () => {
  if (window.confirm("Descartar elimina la alerta: queda fuera del historial consultable y no se indexa. ¿Continuar?")) {
    decide("dismissed");
  }
});

async function loadQueue() {
  try {
    const events = await api("/events/pending-review");
    queue = events.sort((a, b) => a.start_ts.localeCompare(b.start_ts)); // del más antiguo al más reciente
    if (selected && !visibleQueue().some((event) => event.event_id === selected.event_id)) clearSelection();
    renderQueue();
  } catch (error) {
    queueList.replaceChildren(el("li", { class: "empty" }, `No se pudo cargar la cola: ${error.message}`));
  }
}

async function loadFocused(id) {
  try {
    const event = await api(`/events/${encodeURIComponent(id)}`);
    if (event.review_status === "pending") {
      focused = event;
      select(event);
    }
  } catch {
  }
}

try {
  places = Object.fromEntries((await api("/cameras")).filter((c) => c.ubicacion).map((c) => [c.camera_id, c.ubicacion]));
} catch {
}
await loadQueue();
setInterval(loadQueue, QUEUE_REFRESH_MS);

registerView("revision", {
  el: document.querySelector("main.review-layout"),
  onShow: async (id) => {
    await loadQueue();
    if (id) await loadFocused(id);
  },
  onHide: () => video.pause(),
});
