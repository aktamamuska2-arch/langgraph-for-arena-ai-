/* Dashboard transparansi — interaksi minimum, tanpa dependensi. */
(function () {
  "use strict";

  /* ---------- 1. sorot bagian aktif di sidebar ---------- */
  var links = Array.prototype.slice.call(document.querySelectorAll("#nav a"));
  var sections = links
    .map(function (a) { return document.querySelector(a.getAttribute("href")); })
    .filter(Boolean);

  if ("IntersectionObserver" in window && sections.length) {
    var visible = new Map();
    var io = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (e) { visible.set(e.target.id, e.isIntersecting ? e.intersectionRatio : 0); });
        var best = null, bestScore = -1;
        visible.forEach(function (score, id) {
          if (score > bestScore) { bestScore = score; best = id; }
        });
        links.forEach(function (a) {
          a.classList.toggle("active", a.getAttribute("href") === "#" + best);
        });
      },
      { rootMargin: "-84px 0px -55% 0px", threshold: [0, 0.15, 0.4, 0.8] }
    );
    sections.forEach(function (s) { io.observe(s); });
  }

  /* ---------- 2. salin perintah ---------- */
  var toast;
  function notify(msg) {
    if (!toast) {
      toast = document.createElement("div");
      toast.style.cssText =
        "position:fixed;left:50%;bottom:26px;transform:translateX(-50%) translateY(10px);" +
        "background:#12211d;color:#9fe6d2;border:1px solid #1d4d43;border-radius:8px;" +
        "padding:9px 16px;font:12px/1.4 ui-monospace,monospace;z-index:200;opacity:0;" +
        "transition:opacity .18s,transform .18s;pointer-events:none;box-shadow:0 8px 24px rgba(0,0,0,.45)";
      document.body.appendChild(toast);
    }
    toast.textContent = msg;
    toast.style.opacity = "1";
    toast.style.transform = "translateX(-50%) translateY(0)";
    clearTimeout(toast._t);
    toast._t = setTimeout(function () {
      toast.style.opacity = "0";
      toast.style.transform = "translateX(-50%) translateY(10px)";
    }, 1600);
  }

  function copy(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text).then(
        function () { notify("tersalin: " + text.slice(0, 60)); },
        function () { notify("gagal menyalin — pilih teksnya manual"); }
      );
    }
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); notify("tersalin: " + text.slice(0, 60)); }
    catch (err) { notify("gagal menyalin — pilih teksnya manual"); }
    document.body.removeChild(ta);
  }

  document.addEventListener("click", function (ev) {
    var btn = ev.target.closest("button[data-copy]");
    if (btn) { copy(btn.getAttribute("data-copy")); return; }
    var cell = ev.target.closest("td.cmd");
    if (cell) { copy(cell.getAttribute("data-cmd") || cell.textContent.trim()); }
  });

  /* ---------- 3. pencarian cepat di inventaris berkas ---------- */
  var table = document.querySelector("#berkas table");
  if (table && table.tBodies.length) {
    var box = document.createElement("input");
    box.type = "search";
    box.placeholder = "Saring nama berkas…";
    box.setAttribute("aria-label", "Saring nama berkas");
    box.style.cssText =
      "width:100%;margin:6px 0 10px;padding:9px 13px;border-radius:8px;" +
      "border:1px solid #1f2c3a;background:#080d12;color:#e6edf3;" +
      "font:12.5px ui-monospace,monospace;outline:none";
    table.parentNode.insertBefore(box, table);

    box.addEventListener("input", function () {
      var q = box.value.toLowerCase();
      Array.prototype.forEach.call(table.tBodies[0].rows, function (row) {
        row.style.display = row.cells[0].textContent.toLowerCase().indexOf(q) > -1 ? "" : "none";
      });
    });
  }

  /* ---------- 4. buka/tutup semua penampil berkas ---------- */
  var viewers = document.querySelectorAll("details.viewer");
  if (viewers.length > 2) {
    var tools = document.createElement("p");
    tools.className = "small";
    tools.innerHTML =
      '<button data-all="open" style="cursor:pointer;background:#121c26;color:#8fa1b6;' +
      'border:1px solid #1f2c3a;border-radius:6px;padding:4px 10px;font:11px ui-monospace,monospace">buka semua</button> ' +
      '<button data-all="close" style="cursor:pointer;background:#121c26;color:#8fa1b6;' +
      'border:1px solid #1f2c3a;border-radius:6px;padding:4px 10px;font:11px ui-monospace,monospace">tutup semua</button>';
    viewers[0].parentNode.insertBefore(tools, viewers[0]);
    tools.addEventListener("click", function (ev) {
      var btn = ev.target.closest("button[data-all]");
      if (!btn) return;
      var open = btn.getAttribute("data-all") === "open";
      Array.prototype.forEach.call(viewers, function (d) { d.open = open; });
    });
  }
})();
