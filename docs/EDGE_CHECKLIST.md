# Edge checklist

Pre-flight checks for running JuniorPiPython ternary (1.58-bit) inference on edge devices (Pi, Jetson).

- [ ] Model weights are already local; no model pulls.
- [ ] Any local service binds to `127.0.0.1` only.
- [ ] Shared logic comes from JuniorAGI_SDK, not duplicated here.
- [ ] No GitHub Actions workflows exist in this repo.
- [ ] Changes are additive: no deletes, no force-push.

See also: `docs/RAILS.md`, `docs/LOCAL.md`, `docs/NIGHT_CYCLE.md`.
