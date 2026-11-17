import { el } from "./common.js?v=16";

export const VEHICLE_TYPES = [
  ["auto", "Auto"],
  ["moto", "Moto"],
  ["camion", "Camión"],
  ["camioneta", "Camioneta"],
  ["pickup", "Pickup"],
];
export const VEHICLE_REASONS = ["accident", "posible_choque"];

export function vehiclesLabel(vehicles) {
  const names = Object.fromEntries(VEHICLE_TYPES);
  return (vehicles ?? []).map((v) => `${names[v.vehicle_type] ?? v.vehicle_type} ${v.color ?? ""}`.trim()).join("; ");
}

export function createVehicleEditor() {
  const rows = el("div", { class: "vehicle-rows" });

  function addRow(vehicle = {}) {
    const type = el(
      "select",
      { "aria-label": "Tipo de vehículo" },
      el("option", { value: "" }, "Tipo…"),
      ...VEHICLE_TYPES.map(([value, label]) => el("option", { value }, label)),
    );
    type.value = vehicle.vehicle_type ?? "";
    const color = el("input", { type: "text", maxLength: 40, placeholder: "Color", "aria-label": "Color del vehículo", value: vehicle.color ?? "" });
    const row = el(
      "div",
      { class: "vehicle-row" },
      type,
      color,
      el("button", { type: "button", class: "ghost", title: "Quitar vehículo", onclick: () => rows.children.length > 1 && row.remove() }, "✕"),
    );
    rows.append(row);
  }

  const root = el(
    "fieldset",
    { class: "vehicles", hidden: true },
    el("legend", {}, "Vehículos involucrados"),
    rows,
    el("button", { type: "button", class: "ghost", onclick: () => addRow() }, "+ Agregar vehículo"),
    el("p", { class: "hint" }, "Un vehículo por fila, con su tipo y color. Hace falta al menos uno."),
  );

  return {
    root,
    setVehicles: (list) => {
      rows.replaceChildren();
      (list?.length ? list : [{}]).forEach(addRow);
    },
    getVehicles: () =>
      [...rows.children].map((row) => ({
        vehicle_type: row.querySelector("select").value,
        color: row.querySelector("input").value.trim(),
      })),
    setVisible: (visible) => {
      root.hidden = !visible;
    },
    isComplete() {
      const list = this.getVehicles();
      return list.length > 0 && list.every((v) => v.vehicle_type);
    },
  };
}
