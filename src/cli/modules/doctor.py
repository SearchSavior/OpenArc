"""Local Intel Arc diagnostics for `openarc status --doctor`.

Reports the things that silently cost performance or break inference on Arc:
the PCIe link the card actually trained at (vs. what card and slot allow),
the kernel driver in use (xe vs i915), the Intel compute-runtime version, and
render-node group membership. Everything is read from sysfs / the local
package manager; no server or root needed. PCIe link-walking follows
Strata's sycl/serve/xe_telemetry.py.
"""
import glob
import grp
import os
import subprocess


def _read(path: str):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _find_intel_gpus():
    """All Intel DRM devices; discrete Arc cards have a tile0 entry (iGPUs don't)."""
    gpus = []
    for dev in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
        if _read(os.path.join(dev, "vendor")) == "0x8086":
            real = os.path.realpath(dev)
            gpus.append({
                "card": dev.split("/")[-2],
                "path": real,
                "discrete": os.path.isdir(os.path.join(real, "tile0")),
            })
    return gpus


def _driver(gpu_path: str):
    link = os.path.join(gpu_path, "driver")
    if os.path.islink(link):
        return os.path.basename(os.path.realpath(link))
    return None


def _pcie_link(gpu_path: str):
    """(current_gen, max_gen, current_width, max_width) for the card's external link.

    The card's own functions sit behind an internal x1 switch, so the reading
    comes from the first upstream port wider than x1; the max is what card AND
    slot allow (the root port caps it).
    """
    gen = {"2.5": 1, "5.0": 2, "8.0": 3, "16.0": 4, "32.0": 5, "64.0": 6}
    d, chain = gpu_path, []
    while d and d.startswith("/sys/devices/pci") and os.path.exists(os.path.join(d, "current_link_speed")):
        chain.append(d)
        d = os.path.dirname(d)

    def g(path, kind):
        return gen.get((_read(os.path.join(path, f"{kind}_link_speed")) or "").split(" ")[0])

    for port in chain:
        width = int(_read(os.path.join(port, "current_link_width")) or 0)
        if width > 1:
            card_max = g(port, "max")
            slot_max = g(chain[-1], "max")
            max_gen = min(x for x in (card_max, slot_max) if x) if (card_max or slot_max) else None
            max_width_raw = _read(os.path.join(port, "max_link_width"))
            max_width = int(max_width_raw) if max_width_raw else None
            return g(port, "current"), max_gen, width, max_width
    return None, None, None, None


def _compute_runtime_version():
    """Intel compute-runtime (Level Zero) version from the local package manager."""
    for cmd in (["pacman", "-Q", "intel-compute-runtime"],
                ["dpkg-query", "-W", "-f=${Version}", "intel-level-zero-gpu"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip().split()[-1]
        except (OSError, subprocess.SubprocessError):
            continue
    return None


def _render_access():
    """(node, group, accessible) for each render node."""
    gids = set(os.getgroups()) | {os.getgid()}
    rows = []
    for node in sorted(glob.glob("/dev/dri/renderD*")):
        try:
            st = os.stat(node)
        except OSError:
            continue
        try:
            group = grp.getgrgid(st.st_gid).gr_name
        except KeyError:
            group = str(st.st_gid)
        rows.append((node, group, os.access(node, os.R_OK | os.W_OK)))
    return rows


def collect_diagnostics():
    """Gather all doctor checks. Returns a dict; never raises on missing data."""
    gpus = _find_intel_gpus()
    for gpu in gpus:
        gpu["driver"] = _driver(gpu["path"])
        cur_gen, max_gen, width, max_width = _pcie_link(gpu["path"])
        gpu["pcie"] = {"gen": cur_gen, "gen_max": max_gen, "width": width, "width_max": max_width}
    return {
        "gpus": gpus,
        "compute_runtime": _compute_runtime_version(),
        "render_nodes": _render_access(),
    }


def evaluate(diag):
    """Turn diagnostics into (level, message) findings. level: ok / warn."""
    findings = []
    if not diag["gpus"]:
        findings.append(("warn", "no Intel GPU found under /sys/class/drm"))
        return findings
    for gpu in diag["gpus"]:
        label = "discrete Arc" if gpu["discrete"] else "integrated GPU"
        driver = gpu["driver"] or "unknown"
        if gpu["discrete"] and driver == "i915":
            findings.append(("warn", f"{gpu['card']} ({label}) is bound to i915; Arc discrete cards want the xe driver"))
        else:
            findings.append(("ok", f"{gpu['card']} ({label}) driver: {driver}"))
        pcie = gpu["pcie"]
        if gpu["discrete"] and pcie["gen"]:
            cur = f"Gen{pcie['gen']} x{pcie['width']}"
            best = f"Gen{pcie['gen_max']} x{pcie['width_max']}" if pcie["gen_max"] else "unknown"
            if (pcie["gen_max"] and pcie["gen"] < pcie["gen_max"]) or \
               (pcie["width_max"] and pcie["width"] < pcie["width_max"]):
                findings.append(("warn", f"{gpu['card']} PCIe link trained at {cur}, but card and slot allow {best}; "
                                         "prefill and model loads are link-bound (reseat, riser, or BIOS link settings)"))
            else:
                findings.append(("ok", f"{gpu['card']} PCIe link: {cur} (max {best})"))
    rt = diag["compute_runtime"]
    if rt:
        findings.append(("ok", f"Intel compute-runtime: {rt}"))
    else:
        findings.append(("warn", "Intel compute-runtime (Level Zero) not found via pacman/dpkg"))
    for node, group, accessible in diag["render_nodes"]:
        if accessible:
            findings.append(("ok", f"{node}: group '{group}', accessible"))
        else:
            findings.append(("warn", f"{node}: group '{group}', NOT accessible by this user "
                                     "(add the user to the render group)"))
    return findings
