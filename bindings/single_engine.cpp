// single_engine.cpp — unified single-node engine module
// ============================================================================
// One import for everything single-node:
//   * build/update  — DiskANNIndex{,UInt8,Int8}: build, insert/delete,
//                     consolidate, save/load (from diskann_bindings.cpp)
//   * search        — SingleNodeSearch{Float,UInt8,Int8} (= PQSearchHarness*):
//                     beam search with search_dist ∈ {exact, pq[_rerank],
//                     sq8[_rerank], rabitq[_rerank]}, rerank_scope ∈ {full,beam},
//                     PQ filter, partition-aware local/remote fetch accounting
//                     (from pq_search_bindings.cpp)
// The legacy modules (diskann_bindings, pq_search_bindings) are still built from
// the same sources for backward compatibility; classes are py::module_local().
// ============================================================================

#include <pybind11/pybind11.h>

namespace py = pybind11;

void register_diskann_build(py::module_& m);   // diskann_bindings.cpp
void register_single_search(py::module_& m);   // pq_search_bindings.cpp

PYBIND11_MODULE(single_engine, m) {
    m.doc() = "SparsePath single-node engine: graph build/update + single-node "
              "search experiments (quantized search modes, PQ filter, fetch accounting).";
    register_diskann_build(m);
    register_single_search(m);
}
