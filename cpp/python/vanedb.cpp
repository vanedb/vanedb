#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#if defined(__AVX__) || defined(__AVX2__) || defined(__FMA__)
#error "The Python extension must be compiled for baseline x86-64"
#endif

#include "core/approx_index.h"
#include "core/flat_index.h"
#include "core/disk_index.h"
#include "core/version.h"

namespace py = pybind11;
using namespace vanedb;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

/// Validates a 1-D float array's shape and copies it into owned storage.
///
/// The copy is the point. `py::array_t<..., c_style | forcecast>` does not
/// copy an input that is already float32 and C-contiguous, so `buf.ptr`
/// aliases the caller's numpy array. Reading through that pointer after
/// `gil_scoped_release` lets another Python thread mutate the data part-way
/// through a scan or graph walk -- and because the finite check runs inside
/// the core, after the release, a value can pass validation and then become
/// NaN before it is stored, defeating the invariant the non-finite
/// conformance suite exists to protect (vanedb#95).
///
/// PyO3 copies into owned `Vec`s before `py.detach` at every site; this keeps
/// the two Python packages on one memory-safety discipline.
std::vector<float> owned_vector(const FloatArray& array, size_t expected_dim,
                                const char* ndim_message, const char* dim_message) {
  py::buffer_info buf = array.request();
  if (buf.ndim != 1) {
    throw std::runtime_error(ndim_message);
  }
  if (static_cast<size_t>(buf.size) != expected_dim) {
    throw std::runtime_error(dim_message);
  }
  const float* first = static_cast<const float*>(buf.ptr);
  return std::vector<float>(first, first + buf.size);
}

}  // namespace

PYBIND11_MODULE(vanedb_cpp, m) {
    m.doc() = "VaneDB - Embeddable vector database for edge AI";

    // Version info
    m.attr("__version__") = VERSION_STRING;
    m.attr("VERSION_MAJOR") = VERSION_MAJOR;
    m.attr("VERSION_MINOR") = VERSION_MINOR;
    m.attr("VERSION_PATCH") = VERSION_PATCH;

    m.def("simd_backend", []() { return detail::runtime_kernels().name; },
          "Active distance backend: scalar, neon, or avx2_fma (CPU/OS checked).");

    py::enum_<Metric>(m, "Metric")
        .value("L2", Metric::L2)
        .value("COSINE", Metric::COSINE)
        .value("DOT", Metric::DOT)
        ;

    // Deprecated alias for backward compatibility.

    // Bind ApproxIndex class
    py::class_<ApproxIndex>(m, "ApproxIndex")
        .def(py::init<size_t, Metric, size_t, size_t, size_t, uint32_t>(),
             py::arg("dimension"),
             py::arg("metric") = Metric::L2,
             py::arg("max_elements") = 100000,
             py::arg("M") = 16,
             py::arg("ef_construction") = 200,
             py::arg("random_seed") = 42)
        .def("add", [](ApproxIndex& self, uint64_t id, py::array_t<float, py::array::c_style | py::array::forcecast> vector_array) {
                const std::vector<float> owned =
                    owned_vector(vector_array, self.dimension(), "Vector must be a 1-dimensional array", "Vector dimension mismatch");
                // Release GIL during potentially long C++ operation
                py::gil_scoped_release release;
                self.add(id, owned.data());
            },
            py::arg("id"), py::arg("vector"),
            "Adds a vector to the index")
        .def("get_vector", [](const ApproxIndex& self, uint64_t id) {
                std::vector<float> vec = self.get_vector(id);
                // Create array that owns its data by using a capsule to prevent use-after-free
                auto* vec_ptr = new std::vector<float>(std::move(vec));
                auto capsule = py::capsule(vec_ptr, [](void* p) {
                    delete static_cast<std::vector<float>*>(p);
                });
                return py::array_t<float>(
                    {vec_ptr->size()},
                    {sizeof(float)},
                    vec_ptr->data(),
                    capsule  // Prevents deallocation until array is destroyed
                );
            },
            py::arg("id"),
            "Retrieves the vector associated with the given ID as a numpy array")
        .def("search", [](const ApproxIndex& self, py::array_t<float, py::array::c_style | py::array::forcecast> query_array, size_t k) {
                const std::vector<float> owned =
                    owned_vector(query_array, self.dimension(), "Query vector must be a 1-dimensional array", "Query dimension mismatch");

                std::vector<HNSWSearchResult> results;
                {
                    // Release GIL during search
                    py::gil_scoped_release release;
                    results = self.search(owned.data(), k);
                }

                // Create numpy arrays for IDs and distances (requires GIL)
                py::array_t<uint64_t> ids(static_cast<py::ssize_t>(results.size()));
                py::array_t<float> dists(static_cast<py::ssize_t>(results.size()));

                auto ids_ptr = ids.mutable_unchecked<1>();
                auto dists_ptr = dists.mutable_unchecked<1>();

                for (size_t i = 0; i < results.size(); ++i) {
                    ids_ptr(i) = results[i].id;
                    dists_ptr(i) = results[i].distance;
                }

                return py::make_tuple(ids, dists);
            },
            py::arg("query_vector"), py::arg("k"),
            "Searches for k nearest neighbors. Returns a tuple (ids, distances).")
        .def("size", &ApproxIndex::size, "Returns the number of vectors in the index")
        .def("dimension", &ApproxIndex::dimension, "Returns the dimension of stored vectors")
        .def("capacity", &ApproxIndex::capacity, "Returns the maximum capacity of the index")
        .def("contains", &ApproxIndex::contains, py::arg("id"), "Checks if a vector with given ID exists")
        .def("set_ef_search", &ApproxIndex::set_ef_search, py::arg("ef"), "Sets the ef parameter for search")
        .def("get_ef_search", &ApproxIndex::get_ef_search, "Returns the current ef_search parameter")
        .def("save", [](const ApproxIndex& self, const std::string& filename) {
                py::gil_scoped_release release;
                self.save(filename);
            },
            py::arg("filename"), "Saves the index to a binary file")
        .def_static("load", [](const std::string& filename) {
                py::gil_scoped_release release;
                return ApproxIndex::load(filename);
            },
            py::arg("filename"), "Loads the index from a binary file",
            py::return_value_policy::take_ownership);

    // Bind FlatIndex class (brute-force, thread-safe)
    py::class_<FlatIndex>(m, "FlatIndex")
        .def(py::init<size_t, Metric>(),
             py::arg("dimension"),
             py::arg("metric") = Metric::L2,
             "Creates a new in-memory vector store")
        .def("add", [](FlatIndex& self, uint64_t id, py::array_t<float, py::array::c_style | py::array::forcecast> vector_array) {
                const std::vector<float> owned =
                    owned_vector(vector_array, self.dimension(), "Vector must be a 1-dimensional array", "Vector dimension mismatch");
                py::gil_scoped_release release;
                self.add(id, owned.data());
            },
            py::arg("id"), py::arg("vector"),
            "Adds a vector to the store")
        .def("get", [](const FlatIndex& self, uint64_t id) -> py::object {
                // Use get_copy() for thread-safe copy while holding the lock
                std::vector<float> vec = self.get_copy(id);
                if (vec.empty()) {
                    return py::none();
                }
                // Move the copy into heap-allocated storage for the capsule
                auto* vec_copy = new std::vector<float>(std::move(vec));
                auto capsule = py::capsule(vec_copy, [](void* p) {
                    delete static_cast<std::vector<float>*>(p);
                });
                return py::array_t<float>(
                    {vec_copy->size()},
                    {sizeof(float)},
                    vec_copy->data(),
                    capsule  // Prevents deallocation until array is destroyed
                );
            },
            py::arg("id"),
            "Gets a vector by ID, returns None if not found")
        .def("search", [](const FlatIndex& self, py::array_t<float, py::array::c_style | py::array::forcecast> query_array, size_t k) {
                const std::vector<float> owned =
                    owned_vector(query_array, self.dimension(), "Query vector must be a 1-dimensional array", "Query dimension mismatch");

                std::vector<SearchResult> results;
                {
                    py::gil_scoped_release release;
                    results = self.search(owned.data(), k);
                }

                py::array_t<uint64_t> ids(static_cast<py::ssize_t>(results.size()));
                py::array_t<float> dists(static_cast<py::ssize_t>(results.size()));
                auto ids_ptr = ids.mutable_unchecked<1>();
                auto dists_ptr = dists.mutable_unchecked<1>();

                for (size_t i = 0; i < results.size(); ++i) {
                    ids_ptr(i) = results[i].id;
                    dists_ptr(i) = results[i].distance;
                }

                return py::make_tuple(ids, dists);
            },
            py::arg("query_vector"), py::arg("k"),
            "Searches for k nearest neighbors. Returns (ids, distances).")
        .def("remove", &FlatIndex::remove, py::arg("id"), "Removes a vector by ID")
        .def("update", [](FlatIndex& self, uint64_t id, py::array_t<float, py::array::c_style | py::array::forcecast> vector_array) {
                const std::vector<float> owned =
                    owned_vector(vector_array, self.dimension(), "Vector must be a 1-dimensional array", "Vector dimension mismatch");
                py::gil_scoped_release release;
                return self.update(id, owned.data());
            },
            py::arg("id"), py::arg("vector"),
            "Updates an existing vector")
        .def("size", &FlatIndex::size, "Returns the number of vectors")
        .def("dimension", &FlatIndex::dimension, "Returns the dimension")
        .def("contains", &FlatIndex::contains, py::arg("id"), "Checks if ID exists")
        .def("clear", &FlatIndex::clear, "Removes all vectors")
        .def("reserve", &FlatIndex::reserve, py::arg("capacity"), "Pre-allocates space");

    // Bind DiskIndexBuilder class
    py::class_<DiskIndexBuilder>(m, "DiskIndexBuilder")
        .def(py::init<size_t, Metric>(),
             py::arg("dimension"),
             py::arg("metric") = Metric::L2,
             "Creates a new builder for memory-mapped vector store")
        .def("add", [](DiskIndexBuilder& self, uint64_t id, py::array_t<float, py::array::c_style | py::array::forcecast> vector_array) {
                const std::vector<float> owned =
                    owned_vector(vector_array, self.dimension(), "Vector must be a 1-dimensional array", "Vector dimension mismatch");
                self.add(id, owned.data());
            },
            py::arg("id"), py::arg("vector"),
            "Adds a vector to the builder")
        .def("save", [](const DiskIndexBuilder& self, const std::string& filename) {
                py::gil_scoped_release release;
                self.save(filename);
            },
            py::arg("filename"),
            "Saves to a memory-mappable file")
        .def("size", &DiskIndexBuilder::size, "Returns the number of vectors")
        .def("dimension", &DiskIndexBuilder::dimension, "Returns the dimension")
        .def("reserve", &DiskIndexBuilder::reserve, py::arg("capacity"), "Pre-allocates space");

    // Bind DiskIndex class (read-only, memory-mapped)
    py::class_<DiskIndex>(m, "DiskIndex")
        .def(py::init<const std::string&>(),
             py::arg("filename"),
             "Opens a memory-mapped vector store file")
        .def("get", [](const DiskIndex& self, uint64_t id) -> py::object {
                const float* ptr = self.get(id);
                if (ptr == nullptr) {
                    return py::none();
                }
                // Keep a zero-copy view and retain the mapping's owner.
                auto view = py::array_t<float>(
                    {self.dimension()},
                    {sizeof(float)},
                    ptr,
                    py::cast(&self)  // Keep store alive while array exists
                );
                // A const pointer does not make a NumPy array read-only.
                // Match the OS mapping protection before exposing any view.
                view.attr("setflags")(py::arg("write") = false);
                return view;
            },
            py::arg("id"),
            "Gets a read-only zero-copy vector from the mapped file, or None if not found. "
            "The array keeps the mapping alive; use .copy() for an editable array.")
        .def("search", [](const DiskIndex& self, py::array_t<float, py::array::c_style | py::array::forcecast> query_array, size_t k) {
                const std::vector<float> owned =
                    owned_vector(query_array, self.dimension(), "Query vector must be a 1-dimensional array", "Query dimension mismatch");

                std::vector<SearchResult> results;
                {
                    py::gil_scoped_release release;
                    results = self.search(owned.data(), k);
                }

                py::array_t<uint64_t> ids(static_cast<py::ssize_t>(results.size()));
                py::array_t<float> dists(static_cast<py::ssize_t>(results.size()));
                auto ids_ptr = ids.mutable_unchecked<1>();
                auto dists_ptr = dists.mutable_unchecked<1>();

                for (size_t i = 0; i < results.size(); ++i) {
                    ids_ptr(i) = results[i].id;
                    dists_ptr(i) = results[i].distance;
                }

                return py::make_tuple(ids, dists);
            },
            py::arg("query_vector"), py::arg("k"),
            "Searches for k nearest neighbors. Returns (ids, distances).")
        .def("size", &DiskIndex::size, "Returns the number of vectors")
        .def("dimension", &DiskIndex::dimension, "Returns the dimension")
        .def("contains", &DiskIndex::contains, py::arg("id"), "Checks if ID exists");
}
