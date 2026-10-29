# lits_build_tree — graph-build-side DiskANN bindings (LITS-aware)

Snapshot of branch `origin/lits-updatecycle` (`e7c07bf90e33`) of the SparsePath repo, C++ only.
It differs from the search-side tree at the package root in exactly three files:

* `bindings/diskann_bindings.cpp` (+194)  — `enable_lits`, capacity-balanced placement, tau knobs
* `third_party/diskann/include/index.h` (+66)
* `third_party/diskann/src/index.cpp` (+699) — LITS edge filter at insert and at delete-consolidation prune

Built by `../../build_lits.sh` into `lib/diskann_bindings.so`; consumed only by
`tools/graph_build/build_updatecycle_lits.py --bindings-lib third_party/lits_build_tree/lib`.
The distributed search engine never loads this tree.
