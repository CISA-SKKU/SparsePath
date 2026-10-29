#pragma once
// core/priority_queue.h — VIDNeighborPriorityQueue
// Extracted from: distributed_bindings.cpp L234-323
// Port of diskann::NeighborPriorityQueue with uint64_t VectorIDs

#include <cstdint>
#include <cstring>
#include <vector>
#include <unordered_set>
#include <utility>

namespace danns {

struct VIDNeighbor {
    uint64_t vid;
    float distance;
    bool expanded;

    VIDNeighbor() = default;
    VIDNeighbor(uint64_t vid, float distance) : vid(vid), distance(distance), expanded(false) {}

    inline bool operator<(const VIDNeighbor& other) const {
        return distance < other.distance || (distance == other.distance && vid < other.vid);
    }
    inline bool operator==(const VIDNeighbor& other) const {
        return vid == other.vid;
    }
};

class VIDNeighborPriorityQueue {
  public:
    VIDNeighborPriorityQueue() : _size(0), _capacity(0), _cur(0) {}
    explicit VIDNeighborPriorityQueue(size_t capacity)
        : _size(0), _capacity(capacity), _cur(0), _data(capacity + 1) {}

    void enable_pool_recording() {
        _record_pool = true;
        _pool.reserve(_capacity * 10);
        _pool_seen.reserve(_capacity * 10);
    }

    const std::vector<std::pair<uint64_t, float>>& get_pool() const { return _pool; }

    bool insert(const VIDNeighbor& nbr, uint64_t* evicted_vid = nullptr) {
        if (_size == _capacity && _data[_size - 1] < nbr)
            return false;

        size_t lo = 0, hi = _size;
        while (lo < hi) {
            size_t mid = (lo + hi) >> 1;
            if (nbr < _data[mid]) {
                hi = mid;
            } else if (_data[mid].vid == nbr.vid) {
                return false;  // duplicate
            } else {
                lo = mid + 1;
            }
        }

        if (_size == _capacity && evicted_vid) {
            *evicted_vid = _data[_size - 1].vid;
        }

        if (lo < _capacity) {
            std::memmove(&_data[lo + 1], &_data[lo], (_size - lo) * sizeof(VIDNeighbor));
        }
        _data[lo] = {nbr.vid, nbr.distance};
        if (_size < _capacity) _size++;
        if (lo < _cur) _cur = lo;
        return true;
    }

    VIDNeighbor closest_unexpanded() {
        _data[_cur].expanded = true;
        size_t pre = _cur;
        if (_record_pool && _pool_seen.insert(_data[pre].vid).second) {
            _pool.push_back({_data[pre].vid, _data[pre].distance});
        }
        while (_cur < _size && _data[_cur].expanded)
            _cur++;
        return _data[pre];
    }

    bool has_unexpanded_node() const { return _cur < _size; }
    size_t size() const { return _size; }
    size_t capacity() const { return _capacity; }

    VIDNeighbor& operator[](size_t i) { return _data[i]; }
    VIDNeighbor operator[](size_t i) const { return _data[i]; }

    void clear() { _size = 0; _cur = 0; }

  private:
    size_t _size, _capacity, _cur;
    bool _record_pool = false;
    std::vector<std::pair<uint64_t, float>> _pool;
    std::unordered_set<uint64_t> _pool_seen;
    std::vector<VIDNeighbor> _data;
};

} // namespace danns
