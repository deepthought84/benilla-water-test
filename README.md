# benilla · Improved Water

A fork of [benilla](https://github.com/samwhosung/benilla), the World of Warcraft 1.12.1 client
written from scratch in Rust and Bevy, that adds **Improved Water**: an optional water look for the
client's lakes, rivers, falls and sea.

## Improved Water

- **Off by default.** Unchecked, water draws the way the 1.12 client does.
- **On:** a rippling surface with a moving glint, a sky sheen at glancing angles, foam along the
  shoreline, and each zone's own water colours following the time of day.
- **Reflections:** lakes, rivers and the sea reflect as planar mirrors; falls, rapids and small
  pools reflect cube probes captured from fixed spots on the water.
- **Classification:** each map's water is sorted into mirror and probe water once, on the map's
  first load (a few seconds), and cached in `benilla-config/Cache/`.
- **A separate module:** the look lives in its own crate, `crates/benilla-water`, plugged into the
  client through a small seam (`crates/benilla-world/src/liquid/hooks.rs`). Built without it, the
  client draws the reference water unchanged.

## Building and running

You need stable Rust and your own World of Warcraft 1.12.1 install.

```sh
WOW_DATA="/path/to/World of Warcraft 1.12.1/Data" cargo run --release -p benilla
```

- Turn the look on in the options window (**Improved Water**), or for one session with
  `WOW_WATER_STYLE=1`.
- A player build, without the developer tools:
  `cargo build --release -p benilla --no-default-features --features improved-water`
- In a developer build, `Ctrl+Shift+D` opens the debug panel. Its Water section switches
  screen-space reflection on (off by default), turns the probes off, and shows the probe spots on
  the minimap and in the world.

This fork is developed and tested on Linux.

## Upstream

Everything about the client itself (what it plays, how it is built, how to contribute) is in the
original repository: <https://github.com/samwhosung/benilla>.

## License

Licensed, like benilla, under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option. No game data is included: the client reads your own 1.12.1 install. World of Warcraft
is a trademark of Blizzard Entertainment; this project is not affiliated with Blizzard.
