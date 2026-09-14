<div align="center">
  <h1>benilla</h1>
  <p><b>A from-scratch World of Warcraft 1.12.1 client in Rust and <a href="https://bevy.org">Bevy</a></b></p>
  <p>
    <a href="https://discord.gg/wJSJx467G4"><img src="https://img.shields.io/discord/1529280129518538922?style=for-the-badge&logo=discord&logoColor=white&label=discord&color=5865F2" alt="Discord"></a>
    <a href="https://www.youtube.com/playlist?list=PLdCnpZNKxyb8"><img src="https://img.shields.io/badge/devlog-youtube-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="YouTube devlog"></a>
    <a href="LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT%20%2F%20Apache--2.0-blue?style=for-the-badge" alt="License"></a>
  </p>
</div>

> [!IMPORTANT]
> **Issues and pull requests are closed here.** benilla is a solo project developed in a private
> tree; this repo is its export, published as squashed snapshots, so a PR here has nothing to land
> on. The best way to contribute is to join the [Discord](https://discord.gg/wJSJx467G4) and report
> the bugs you find. Questions and ideas are welcome in the same place.

benilla speaks the original 1.12.1 protocol, so it connects to any server the real client could,
and reads its game data at runtime from your own 1.12.1 install. Every file format and the network
protocol are implemented from scratch, with no original client code, no third-party WoW crates,
and no bundled game assets.

It is a **reimplementation**, not an emulator and not a remaster. The target is that a 1.12.1
player cannot tell the difference — which means the reference client's behaviour is the
specification, including the parts that look like bugs.

---

## Running it

You need a **1.12.1 (build 5875) client install** for game data, a vanilla server to connect to,
and stable Rust. Any 1.12.1 core works; [vmangos](https://github.com/vmangos/core) is what
development runs against, and cMaNGOS and the rest speak the same protocol.

```sh
WOW_DATA=/path/to/WoW/Data cargo run --release -p benilla
```

The server defaults to `localhost:3724`, the stock `realmd` auth port. Point `WOW_HOST`
at any IP or hostname, appending the auth port if yours is remapped
(`WOW_HOST=play.example.com:5000`). Credentials go in at the login screen, or set `WOW_USER` /
`WOW_PASS` to skip it.

`WOW_DATA` must point at the **`Data` directory**, not the install root. Without it the client
looks for an install beside the binary and in the project folder.

## What works

- **Formats:** readers for the full asset stack (MPQ patch chain, BLP, DBC, ADT/WDT/WDL, M2, WMO),
  wired into Bevy as an asset source.
- **World:** streamed terrain out to the horizon, portal-culled WMOs with interior lighting,
  doodads and ground clutter, swimmable liquids, sky and weather, and the client's own day/night
  lighting, fog and gamma passes.
- **Models:** GPU-skinned M2s with the full animation controller, a near feature-complete particle
  system, ribbons, and animated gameobjects from doors to lifts.
- **Characters:** customization end to end, the armor texture composite, weapons with sheathing and
  enchant glows, shapeshift forms, stealth and mounts.
- **Movement:** a WoW-feel controller, networked movement in both directions, the server-granted
  modes from slow fall to roots, a follow camera with collision, boats, zeppelins and taxi flights.
- **Networking:** SRP6 auth through world-session crypto, the object mirror into the ECS, and live
  wire coverage from movement and chat through spells, party, quests, mail, trade, vendors, bank,
  loot, the auction house and PvP honor.
- **UI:** a from-scratch FrameXML + Lua engine driving the built-in interface, from the login and
  character screens through the full HUD, the classic windows (guild, macros and key bindings
  included), chat, nameplates, floating combat text and tooltips; third-party addons load from
  a `benilla-config/AddOns/` folder beside the executable (partial: AtlasLoot and Bagnon run).
- **Combat:** melee on the faithful swing law, ranged and Auto Shot, casting with GCD and
  cooldowns, combo points, crowd control that really holds you, and the spell visual pipeline.
- **Audio:** music, ambience and SFX under the client's own selection and crossfade rules, with
  interior and underwater transitions and zone reverb.

## Where it's going

benilla is done when a 1.12.1 player can do everything here that they could in the original
client, it looks and feels the same, and it runs from a download on Windows, Linux and macOS.
No dates; the order is what is likely, not a promise.

- Battlegrounds and meeting stones, then the long tail of small features that separates a
  working client from a finished one.
- Addons, options and performance, ongoing.
- The no-brainer fixes from VanillaFixes, SuperWoW and the like.
- Playable downloads for Windows, Linux and macOS. Linux first.

Not planned: other expansions or client versions, Warden (anticheat).

---

## How it is put together

A Cargo workspace of ~20 crates. The dependency direction is strictly one way: formats know
nothing about rendering, the world knows nothing about the UI, and the binary knows about
everything.

| crate | what it owns |
| --- | --- |
| `benilla` | the binary: plugin wiring and boot order, and nothing else |
| `benilla-app` | the client proper — UI, net session, gameplay systems, capture harness |
| `benilla-ui` | the FrameXML + Lua engine: layout, widgets, the 1.12 API surface |
| `benilla-world` | terrain, WMOs, liquids, sky, weather, lighting, the camera |
| `benilla-formats` | the asset stack, re-exporting the per-format crates below |
| `benilla-protocol` | auth and world protocol, opcodes, the object update mirror |
| `benilla-assets` | Bevy materials and the WGSL shaders they specialize |
| `benilla-mpq` `-blp` `-dbc` `-adt` `-wdt` `-wmo` `-m2` | one file format each |
| `benilla-srp` `-bytes` `-buildstamp` | SRP6, wire primitives, build stamping |
| `benilla-visual` | the image-diff tool the capture harness is graded with |
| `benilla-worldview` | a world viewer that boots without a server |

Rough scale: ~780k lines of Rust, ~6,700 tests, 17 WGSL shaders, and an interface written in the
reference's own XML + Lua so that addons find the names they expect.

## Working on it

Three habits do most of the work here, and the codebase assumes all three.

**The reference is the specification.** Behaviour is reverse-engineered from the real client and
cited where it matters — function addresses, DBC rows, the exact byte a value came from. A comment
saying *why* a constant is what it is, and what was measured to find out, is worth more than the
line it documents; that is why the comments run long. Where the reference and "better" disagree,
the reference wins unless a decision says otherwise in so many words.

**Nothing visual is done until it has been photographed.** There is a deterministic capture rig:

```sh
WOW_CAPTURE=water-noon WOW_CAPTURE_OUT=/tmp/shot.png cargo run --release -p benilla
```

It boots server-less, pins the camera and the game clock, waits for the image to stop changing,
writes one PNG and exits. `WOW_CAPTURE=list` names the golden scenarios; `WOW_CAPTURE=vista` with
`WOW_VISTA_AT=x,y,z` and `WOW_VISTA_FACE=<deg>` puts the camera anywhere, which is how a
screenshot from a player becomes a reproducible shot. `benilla-visual diff-dir` grades two runs.
**Open the PNG and check the subject is in it** — a camera buried in terrain diffs to zero very
convincingly.

**Performance claims are measured, in one sitting.** `WOW_FPS_PROBE=<frames>` prints a
machine-greppable line of frame-time statistics from the same harness; `WOW_FPS_JOURNAL=<csv>`
adds per-second rows with a GPU breakdown per render pass. Both drift between sessions on the
same machine, so **interleave the arms of a comparison inside one batch** rather than measuring
before, rebuilding, and measuring after — two batches twenty minutes apart can differ by more
than the change under test.

### Levers

There are ~400 `WOW_*` environment switches. They are the project's debugging surface: each one is
read in exactly one place, documented where it is read, and most exist because some bug was easier
to see than to reason about. A few that show the shape of the thing:

| lever | what it does |
| --- | --- |
| `WOW_WATER_STYLE=1` | benilla's own water instead of the 1.12 combine |
| `WOW_SSR_SHOW=1` | paint the screen-space march's confidence instead of the water |
| `WOW_MIRROR_SHOW=1` | paint the planar capture the water is about to sample |
| `WOW_MSAA=`, `WOW_FARCLIP=`, `WOW_RENDER_SCALE=` | the graphics dials, without a config file |
| `WOW_CAPTURE=`, `WOW_FPS_PROBE=` | the two harnesses above |

The capture harness is **hermetic**: `benilla-config/config.toml` is not read during a capture, so
every CVar sits at its registered default. A feature behind a non-default CVar does not appear in
the shot at all — which is why the water examples above pass `WOW_WATER_STYLE=1` explicitly.

---

Early inspiration and file format guidance came from the
[wowemulation-dev](https://github.com/wowemulation-dev) community, and
[warcraft-rs](https://github.com/wowemulation-dev/warcraft-rs) in particular.

benilla is an independent fan project, not affiliated with or endorsed by Blizzard Entertainment.
It ships **no Blizzard content** — no art, models, sounds, maps, MPQ contents or FrameXML; you
provide your own legally obtained 1.12.1 client. The interface code under
`crates/benilla-app/assets/ui/` is ours, written to the client's own layout and API names so that
the windows look right and 1.12.1 addons find the names they expect.

World of Warcraft is a trademark of Blizzard Entertainment, Inc. Our own code is licensed under
[MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE), at your option.
