# MEIA Third-Party Notices

This file records third-party software used by or distributed with MEIA 0.11.0. The [PolyForm Noncommercial License 1.0.0](LICENSE) applies only to copyrightable MEIA material owned by Xiaomei_974. It does not replace, narrow, or relicense any third-party component.

## Python environment dependencies

The GitHub source archive declares these packages in `requirements.txt`; users normally install them separately from Python package repositories. Their own distributions carry the authoritative license files and notices.

| Package | Version | License | Upstream license or project |
| --- | ---: | --- | --- |
| ASE | 3.22.1 | LGPL-2.1-or-later | [ASE license information](https://gitlab.com/ase/ase/-/blob/master/doc/development/licenseinfo.rst) |
| Matplotlib | 3.7.3 | PSF-based Matplotlib License plus bundled-component notices | [Matplotlib license](https://github.com/matplotlib/matplotlib/blob/v3.7.3/LICENSE/LICENSE) |
| NumPy | 1.26.4 | BSD-3-Clause plus bundled-component notices | [NumPy license](https://github.com/numpy/numpy/blob/v1.26.4/LICENSE.txt) |
| pandas | 2.3.3 | BSD-3-Clause plus bundled-component notices | [pandas license](https://github.com/pandas-dev/pandas/blob/v2.3.3/LICENSE) |
| Plotly.py | 5.24.1 | MIT | [Plotly.py license](https://github.com/plotly/plotly.py/blob/v5.24.1/LICENSE.txt) |
| Streamlit | 1.38.0 | Apache-2.0 plus bundled-component notices | [Streamlit license](https://github.com/streamlit/streamlit/blob/1.38.0/LICENSE) |
| pytest | 8.3.5 | MIT | [pytest license](https://github.com/pytest-dev/pytest/blob/8.3.5/LICENSE) |

If MEIA is later distributed as a standalone executable, container image, wheel collection, or complete environment, the distributor must audit the actual contents and carry forward every applicable license, notice, and source-code offer. In particular, redistribution of ASE must continue to satisfy LGPL-2.1-or-later.

## Bundled 3D frontend

The repository includes a compiled frontend under `meia/components/atom_viewer/frontend/dist/`. The table below conservatively covers every non-development package in the locked frontend dependency closure. Some type-only or command-line support packages may not contribute executable code to the final browser bundle; they remain listed so that the published notice set does not understate the build inputs.

The archived text in each “License copy” link is copied verbatim from the installed npm package. Apache Arrow's upstream `NOTICE` is preserved separately. `streamlit-component-lib` declares Apache-2.0 but does not include a standalone license file in its npm package, so its archive contains an unmodified copy of the Apache License 2.0 text and the table links to the official Streamlit project license as additional provenance.

| Package | Version | License | License copy |
| --- | ---: | --- | --- |
| `@types/command-line-args` | 5.2.0 | MIT | [text](THIRD_PARTY_LICENSES/frontend/@types__command-line-args@5.2.0.txt) |
| `@types/command-line-usage` | 5.0.2 | MIT | [text](THIRD_PARTY_LICENSES/frontend/@types__command-line-usage@5.0.2.txt) |
| `@types/flatbuffers` | 1.10.3 | MIT | [text](THIRD_PARTY_LICENSES/frontend/@types__flatbuffers@1.10.3.txt) |
| `@types/node` | 18.7.23 | MIT | [text](THIRD_PARTY_LICENSES/frontend/@types__node@18.7.23.txt) |
| `@types/pad-left` | 2.1.1 | MIT | [text](THIRD_PARTY_LICENSES/frontend/@types__pad-left@2.1.1.txt) |
| `ansi-styles` | 3.2.1 | MIT | [text](THIRD_PARTY_LICENSES/frontend/ansi-styles@3.2.1.txt) |
| `apache-arrow` | 11.0.0 | Apache-2.0 and bundled notices | [license](THIRD_PARTY_LICENSES/frontend/apache-arrow@11.0.0.txt), [NOTICE](THIRD_PARTY_LICENSES/frontend/apache-arrow@11.0.0-NOTICE.txt) |
| `array-back` | 3.1.0 | MIT | [text](THIRD_PARTY_LICENSES/frontend/array-back@3.1.0.txt) |
| `chalk` | 2.4.2 | MIT | [text](THIRD_PARTY_LICENSES/frontend/chalk@2.4.2.txt) |
| `color-convert` | 1.9.3 | MIT | [text](THIRD_PARTY_LICENSES/frontend/color-convert@1.9.3.txt) |
| `color-name` | 1.1.3 | MIT | [text](THIRD_PARTY_LICENSES/frontend/color-name@1.1.3.txt) |
| `command-line-args` | 5.2.1 | MIT | [text](THIRD_PARTY_LICENSES/frontend/command-line-args@5.2.1.txt) |
| `command-line-usage` | 6.1.3 | MIT | [text](THIRD_PARTY_LICENSES/frontend/command-line-usage@6.1.3.txt) |
| `array-back` (nested under `command-line-usage`) | 4.0.2 | MIT | [text](THIRD_PARTY_LICENSES/frontend/command-line-usage__node_modules__array-back@4.0.2.txt) |
| `typical` (nested under `command-line-usage`) | 5.2.0 | MIT | [text](THIRD_PARTY_LICENSES/frontend/command-line-usage__node_modules__typical@5.2.0.txt) |
| `deep-extend` | 0.6.0 | MIT | [text](THIRD_PARTY_LICENSES/frontend/deep-extend@0.6.0.txt) |
| `escape-string-regexp` | 1.0.5 | MIT | [text](THIRD_PARTY_LICENSES/frontend/escape-string-regexp@1.0.5.txt) |
| `find-replace` | 3.0.0 | MIT | [text](THIRD_PARTY_LICENSES/frontend/find-replace@3.0.0.txt) |
| `flatbuffers` | 2.0.4 | Apache-2.0 | [text](THIRD_PARTY_LICENSES/frontend/flatbuffers@2.0.4.txt) |
| `has-flag` | 3.0.0 | MIT | [text](THIRD_PARTY_LICENSES/frontend/has-flag@3.0.0.txt) |
| `hoist-non-react-statics` | 3.3.2 | BSD-3-Clause | [text](THIRD_PARTY_LICENSES/frontend/hoist-non-react-statics@3.3.2.txt) |
| `js-tokens` | 4.0.0 | MIT | [text](THIRD_PARTY_LICENSES/frontend/js-tokens@4.0.0.txt) |
| `json-bignum` | 0.0.3 | MIT (license file; package metadata omits identifier) | [text](THIRD_PARTY_LICENSES/frontend/json-bignum@0.0.3.txt) |
| `lodash.camelcase` | 4.3.0 | MIT plus CC0 sample-code notice | [text](THIRD_PARTY_LICENSES/frontend/lodash.camelcase@4.3.0.txt) |
| `loose-envify` | 1.4.0 | MIT | [text](THIRD_PARTY_LICENSES/frontend/loose-envify@1.4.0.txt) |
| `object-assign` | 4.1.1 | MIT | [text](THIRD_PARTY_LICENSES/frontend/object-assign@4.1.1.txt) |
| `pad-left` | 2.1.0 | MIT | [text](THIRD_PARTY_LICENSES/frontend/pad-left@2.1.0.txt) |
| `plotly.js-dist-min` | 2.35.2 | MIT | [text](THIRD_PARTY_LICENSES/frontend/plotly.js-dist-min@2.35.2.txt) |
| `prop-types` | 15.8.1 | MIT | [text](THIRD_PARTY_LICENSES/frontend/prop-types@15.8.1.txt) |
| `react` | 16.14.0 | MIT | [text](THIRD_PARTY_LICENSES/frontend/react@16.14.0.txt) |
| `react-dom` | 16.14.0 | MIT | [text](THIRD_PARTY_LICENSES/frontend/react-dom@16.14.0.txt) |
| `react-is` | 16.13.1 | MIT | [text](THIRD_PARTY_LICENSES/frontend/react-is@16.13.1.txt) |
| `reduce-flatten` | 2.0.0 | MIT | [text](THIRD_PARTY_LICENSES/frontend/reduce-flatten@2.0.0.txt) |
| `repeat-string` | 1.6.1 | MIT | [text](THIRD_PARTY_LICENSES/frontend/repeat-string@1.6.1.txt) |
| `scheduler` | 0.19.1 | MIT | [text](THIRD_PARTY_LICENSES/frontend/scheduler@0.19.1.txt) |
| `streamlit-component-lib` | 2.0.0 | Apache-2.0 | [archived text](THIRD_PARTY_LICENSES/frontend/streamlit-component-lib@2.0.0.txt), [upstream license](https://github.com/streamlit/streamlit/blob/1.38.0/LICENSE) |
| `supports-color` | 5.5.0 | MIT | [text](THIRD_PARTY_LICENSES/frontend/supports-color@5.5.0.txt) |
| `table-layout` | 1.0.2 | MIT | [text](THIRD_PARTY_LICENSES/frontend/table-layout@1.0.2.txt) |
| `array-back` (nested under `table-layout`) | 4.0.2 | MIT | [text](THIRD_PARTY_LICENSES/frontend/table-layout__node_modules__array-back@4.0.2.txt) |
| `typical` (nested under `table-layout`) | 5.2.0 | MIT | [text](THIRD_PARTY_LICENSES/frontend/table-layout__node_modules__typical@5.2.0.txt) |
| `tslib` | 2.8.1 | 0BSD | [text](THIRD_PARTY_LICENSES/frontend/tslib@2.8.1.txt) |
| `typical` | 4.0.0 | MIT | [text](THIRD_PARTY_LICENSES/frontend/typical@4.0.0.txt) |
| `wordwrapjs` | 4.0.1 | MIT | [text](THIRD_PARTY_LICENSES/frontend/wordwrapjs@4.0.1.txt) |
| `typical` (nested under `wordwrapjs`) | 5.2.0 | MIT | [text](THIRD_PARTY_LICENSES/frontend/wordwrapjs__node_modules__typical@5.2.0.txt) |

The Plotly bundle's retained comment points to `plotly.min.js.LICENSE.txt`. An exact source copy is kept at `meia/components/atom_viewer/frontend/public/assets/plotly.min.js.LICENSE.txt`; Vite copies it to `meia/components/atom_viewer/frontend/dist/assets/plotly.min.js.LICENSE.txt` on every build. Both files have the same SHA-256 as the npm package's `LICENSE` file.

## Build-only frontend dependencies

Vite and its build-time dependency closure are not included in the browser bundle solely because they are used to build it. Their metadata remains in `package-lock.json` and their license files are installed by `npm ci`. If future build changes embed any additional component, regenerate and re-audit this notice set before release.

## No endorsement

The names of third-party projects and contributors are used only for attribution. They do not imply sponsorship, affiliation, or endorsement of MEIA.
