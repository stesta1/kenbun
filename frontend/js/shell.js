// Marco común del tablero y la revisión: una sola página, una sola barra superior y una sola conexión.
// Cambiar de vista solo muestra una u oculta la otra, así que lo que está en curso (una consulta al modelo, los canales en
// tiempo real, el sonido) sigue funcionando al pasar de una a otra.
import { bootAuth, mountTopbar } from "./common.js?v=16";

await bootAuth();

export const topbar = mountTopbar(document.getElementById("topbar"));

const views = new Map(); // nombre -> { el, onShow(arg), onHide() }
export const shell = { returnTo: null };
let shown = null;

function parseHash() {
  const [name, arg] = location.hash.replace(/^#/, "").split("/");
  return { name: name === "revision" ? "revision" : "tablero", arg: arg ? decodeURIComponent(arg) : null };
}

function route() {
  const { name, arg } = parseHash();
  const target = views.get(name);
  if (!target) return; // la vista todavía no se registró: lo vuelve a intentar al registrarse
  const key = `${name}/${arg ?? ""}`;
  if (shown === key && !target.el.hidden) return;
  shown = key;
  for (const [other, view] of views) {
    if (other !== name && !view.el.hidden) {
      view.el.hidden = true;
      view.onHide?.();
    }
  }
  if (name === "tablero" || !arg) shell.returnTo = null;
  target.el.hidden = false;
  topbar.setActive(name);
  target.onShow?.(arg);
}

export function registerView(name, view) {
  views.set(name, view);
  route();
}

export function goTo(name) {
  location.hash = `#${name}`;
}

export function openReview(eventId) {
  shell.returnTo = "tablero";
  location.hash = `#revision/${encodeURIComponent(eventId)}`;
}

window.addEventListener("hashchange", route);
