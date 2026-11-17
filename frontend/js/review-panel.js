import { SELECTABLE_CATEGORIES, api, el, fmtDateTime, reasonLabel, resolveClipUrl } from "./common.js?v=16";
import { VEHICLE_REASONS, createVehicleEditor } from "./vehicles.js?v=16";

export function createReviewPanel(root, { onResolved, places = () => ({}) }) {
  const video = el("video", { controls: true, playsinline: true, class: "review-video" });
  const meta = el("dl", { class: "meta" });
  const category = el(
    "select",
    { "aria-label": "Categoría del evento" },
    ...SELECTABLE_CATEGORIES.map((reason) => el("option", { value: reason }, reasonLabel(reason))),
  );
  const description = el("textarea", { rows: 6, "aria-label": "Descripción del evento" });
  const vehicles = createVehicleEditor();
  const status = el("p", { role: "status" });
  const confirmButton = el("button", { type: "button", class: "primary" }, "Confirmar evento");
  const dismissButton = el("button", { type: "button", class: "danger" }, "Descartar evento");
  const closeButton = el("button", { type: "button", class: "ghost", title: "Cierra el panel sin decidir; la alerta sigue en la lista" }, "Cerrar");
  const actions = [confirmButton, dismissButton];

  root.replaceChildren(
    el("div", { class: "panel-head" }, el("h2", {}, "Revisar evento"), el("span", { class: "spacer" }), closeButton),
    el(
      "div",
      { class: "review-body" },
      video,
      meta,
      el("label", { class: "field" }, "Categoría del evento", category),
      el("p", { class: "hint" }, "Si la cambiás, al confirmar el evento queda con la categoría nueva."),
      description,
      vehicles.root,
      el("div", { class: "decision-buttons" }, confirmButton, dismissButton),
      el("p", { class: "hint" }, "Confirmar guarda la descripción y los vehículos y deja el evento consultable. Descartar no lo indexa."),
      status,
    ),
  );

  let current = null;

  function say(message, isError = false) {
    status.textContent = message;
    status.className = isError ? "error" : "";
  }

  function syncVehicles() {
    vehicles.setVisible(VEHICLE_REASONS.includes(category.value));
  }

  function open(event) {
    current = event;
    say("");
    root.hidden = false;
    resolveClipUrl(event.clip_path)
      .then((url) => {
        if (current !== event) return;
        video.src = url;
        return video.play();
      })
      .catch(() => {});
    const place = places()[event.camera_id];
    const rows = [
      ["Motivo", reasonLabel(event.trigger_reason)],
      ["Cámara", event.camera_label],
      ["Ubicación", place ?? "—"],
      ["Inicio", fmtDateTime(event.start_ts)],
      ["Confianza", event.confidence === null ? "—" : `${Math.round(event.confidence * 100)} %`],
    ];
    meta.replaceChildren(...rows.flatMap(([term, value]) => [el("dt", {}, term), el("dd", {}, value)]));
    category.value = event.trigger_reason;
    description.value = event.description_nl ?? "";
    vehicles.setVehicles(event.vehicles);
    syncVehicles();
    actions.forEach((button) => (button.disabled = false));
    const waiting = !event.description_nl && event.status !== "failed";
    confirmButton.disabled = waiting;
    if (waiting) say("La descripción todavía se está generando.");
  }

  function close() {
    current = null;
    video.pause();
    video.removeAttribute("src");
    video.load();
    root.hidden = true;
  }

  async function decide(action) {
    if (!current) return;
    const event = current;
    const text = description.value.trim();
    const needsVehicles = VEHICLE_REASONS.includes(category.value);
    if (action === "confirmed") {
      if (!text) return say("Escribí la descripción antes de confirmar.", true);
      if (needsVehicles && !vehicles.isComplete()) return say("Un choque necesita al menos un vehículo, con su tipo.", true);
    }
    actions.forEach((button) => (button.disabled = true));
    try {
      if (action === "confirmed") {
        await api(`/events/${event.event_id}/details`, {
          method: "POST",
          body: {
            description_nl: text,
            trigger_reason: category.value !== event.trigger_reason ? category.value : null,
            vehicles: needsVehicles ? vehicles.getVehicles() : [],
          },
        });
      } else {
        await api(`/events/${event.event_id}/review`, { method: "POST", body: { review_status: "dismissed" } });
      }
      close();
      onResolved(event);
    } catch (error) {
      say(`No se pudo guardar: ${error.message}`, true);
      actions.forEach((button) => (button.disabled = false));
    }
  }

  category.addEventListener("change", syncVehicles);
  description.addEventListener("input", () => {
    if (current && description.value.trim()) confirmButton.disabled = false;
  });
  confirmButton.addEventListener("click", () => decide("confirmed"));
  dismissButton.addEventListener("click", () => {
    if (window.confirm("Descartar elimina la alerta: queda fuera del historial consultable y no se indexa. ¿Continuar?")) decide("dismissed");
  });
  closeButton.addEventListener("click", close);

  return { open, close, isOpen: () => current !== null };
}
