#include "gguf/gguf.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>

#include "nibble/error.h"
#include "util/escape.h"

namespace nibble::gguf {
namespace {

constexpr uint32_t kDefaultAlignment = 32;
constexpr uint64_t kMaxI64 = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());

class Reader {
 public:
  explicit Reader(std::span<const uint8_t> b) : bytes_(b) {}

  size_t pos() const { return pos_; }
  size_t remaining() const { return bytes_.size() - pos_; }

  template <class T>
  T read(const char* what) {
    if (remaining() < sizeof(T)) fail_eof(what);
    T v;
    std::memcpy(&v, bytes_.data() + pos_, sizeof v);  // little-endian host assumed
    pos_ += sizeof v;
    return v;
  }

  const uint8_t* here() const { return bytes_.data() + pos_; }
  void skip(size_t n) { pos_ += n; }  // caller checked n <= remaining()
  std::span<const uint8_t> span_from(size_t start) const { return bytes_.subspan(start, pos_ - start); }
  void skip_string(const char* what) {
    uint64_t len = read<uint64_t>(what);
    if (len > remaining()) {
      throw Error(std::string("gguf: string length ") + std::to_string(len) + " exceeds file size while reading " + what);
    }
    pos_ += static_cast<size_t>(len);
  }

  std::string read_string(const char* what) {
    uint64_t len = read<uint64_t>(what);
    if (len > remaining()) {
      throw Error(std::string("gguf: string length ") + std::to_string(len) + " exceeds file size while reading " + what);
    }
    std::string s(reinterpret_cast<const char*>(bytes_.data() + pos_), static_cast<size_t>(len));
    pos_ += static_cast<size_t>(len);
    return s;
  }

 private:
  [[noreturn]] void fail_eof(const char* what) const {
    throw Error(std::string("gguf: unexpected end of file while reading ") + what);
  }
  std::span<const uint8_t> bytes_;
  size_t pos_ = 0;
};

// Wire size of a fixed-size scalar type; 0 for String/Array/unknown.
size_t scalar_size(ValueType t) {
  switch (t) {
    case ValueType::U8: case ValueType::I8: case ValueType::Bool: return 1;
    case ValueType::U16: case ValueType::I16: return 2;
    case ValueType::U32: case ValueType::I32: case ValueType::F32: return 4;
    case ValueType::U64: case ValueType::I64: case ValueType::F64: return 8;
    default: return 0;
  }
}

ValueType read_value_type(Reader& r, const std::string& key) {
  uint32_t t = r.read<uint32_t>("value type");
  if (t > static_cast<uint32_t>(ValueType::F64)) {
    throw Error("gguf: key '" + escape_control(key) + "' has unknown value type " + std::to_string(t));
  }
  return static_cast<ValueType>(t);
}

// Reads one integer/float scalar. Integers go to `i`, floats to `f`.
void read_scalar(Reader& r, ValueType t, const std::string& key, int64_t& i, double& f, bool& is_float) {
  is_float = false;
  switch (t) {
    case ValueType::U8: i = r.read<uint8_t>("u8"); break;
    case ValueType::I8: i = r.read<int8_t>("i8"); break;
    case ValueType::U16: i = r.read<uint16_t>("u16"); break;
    case ValueType::I16: i = r.read<int16_t>("i16"); break;
    case ValueType::U32: i = r.read<uint32_t>("u32"); break;
    case ValueType::I32: i = r.read<int32_t>("i32"); break;
    case ValueType::Bool: i = r.read<uint8_t>("bool") != 0; break;
    case ValueType::I64: i = r.read<int64_t>("i64"); break;
    case ValueType::U64: {
      uint64_t v = r.read<uint64_t>("u64");
      if (v > kMaxI64) throw Error("gguf: key '" + escape_control(key) + "' u64 value " + std::to_string(v) + " exceeds int64 range");
      i = static_cast<int64_t>(v);
      break;
    }
    case ValueType::F32: f = r.read<float>("f32"); is_float = true; break;
    case ValueType::F64: f = r.read<double>("f64"); is_float = true; break;
    default: throw Error("gguf: key '" + escape_control(key) + "' internal: not a scalar type");
  }
}

}  // namespace

// Arrays are recorded as views into the file bytes (no per-element decode or
// allocation beyond one 8-byte offset per string element), so memory stays
// bounded by the file size even for adversarial element counts.
struct detail::ArrayAccess {
  static Array read(Reader& r, const std::string& key);
};

Array detail::ArrayAccess::read(Reader& r, const std::string& key) {
  Array a;
  a.elem_type_ = read_value_type(r, key);
  if (a.elem_type_ == ValueType::Array) throw Error("gguf: key '" + escape_control(key) + "' is a nested array (unsupported)");
  uint64_t count = r.read<uint64_t>("array count");
  uint64_t min_elem = a.elem_type_ == ValueType::String ? 8 : scalar_size(a.elem_type_);
  uint64_t need;
  if (__builtin_mul_overflow(count, min_elem, &need) || need > r.remaining()) {
    throw Error("gguf: key '" + escape_control(key) + "' array count " + std::to_string(count) + " exceeds file size");
  }
  const size_t n = static_cast<size_t>(count);
  a.count_ = n;
  const size_t start = r.pos();
  if (a.elem_type_ == ValueType::String) {
    a.str_off_.reserve(n);  // n * 8 <= remaining bytes (checked above)
    for (size_t k = 0; k < n; ++k) {
      a.str_off_.push_back(r.pos() - start);
      r.skip_string("array string");
    }
  } else {
    if (a.elem_type_ == ValueType::U64) {  // keep the int64 contract of int_at()
      for (size_t k = 0; k < n; ++k) {
        uint64_t v;
        std::memcpy(&v, r.here() + k * 8, 8);
        if (v > kMaxI64) {
          throw Error("gguf: key '" + escape_control(key) + "' u64 element " + std::to_string(k) + " value " + std::to_string(v) +
                      " exceeds int64 range");
        }
      }
    }
    r.skip(static_cast<size_t>(need));
  }
  a.data_ = r.span_from(start);
  return a;
}

namespace {

Value read_value(Reader& r, ValueType t, const std::string& key) {
  if (t == ValueType::String) return r.read_string("string value");
  if (t == ValueType::Array) return detail::ArrayAccess::read(r, key);
  int64_t i = 0;
  double f = 0;
  bool is_float;
  read_scalar(r, t, key, i, f, is_float);
  if (is_float) return f;
  if (t == ValueType::Bool) return i != 0;
  return i;
}

uint64_t align_up(uint64_t v, uint64_t a, const char* what) {
  uint64_t s;
  if (__builtin_add_overflow(v, a - 1, &s)) throw Error(std::string("gguf: overflow aligning ") + what);
  return s / a * a;
}

const char* type_label(const Value& v) {
  switch (v.index()) {
    case 0: return "integer";
    case 1: return "float";
    case 2: return "bool";
    case 3: return "string";
    default: return "array";
  }
}

}  // namespace

File File::open(const std::string& path) {
  try {
    MappedFile m = MappedFile::open(path);
    File f = parse_impl(m.bytes());
    f.map_ = std::move(m);  // moves the handle; the mapping address (and every view into it) is unchanged
    return f;
  } catch (const std::bad_alloc&) {
    throw Error("out of memory loading " + path);
  }
}

File File::parse(std::span<const uint8_t> bytes) {
  try {
    return parse_impl(bytes);
  } catch (const std::bad_alloc&) {
    throw Error("out of memory loading GGUF data");
  }
}

File File::parse_impl(std::span<const uint8_t> bytes) {
  Reader r(bytes);
  File f;

  uint32_t magic = r.read<uint32_t>("magic");
  if (magic != 0x46554747u) throw Error("gguf: bad magic (not a GGUF file)");
  f.version_ = r.read<uint32_t>("version");
  if (f.version_ != 2 && f.version_ != 3) {
    throw Error("gguf: unsupported version " + std::to_string(f.version_) + " (supported: 2, 3)");
  }
  uint64_t n_tensors = r.read<uint64_t>("tensor count");
  uint64_t n_kv = r.read<uint64_t>("kv count");
  if (n_kv > r.remaining() / 12) throw Error("gguf: kv count " + std::to_string(n_kv) + " exceeds file size");
  if (n_tensors > r.remaining() / 24) throw Error("gguf: tensor count " + std::to_string(n_tensors) + " exceeds file size");

  f.metadata_.reserve(std::min<uint64_t>(n_kv, 4096));  // count is untrusted
  for (uint64_t k = 0; k < n_kv; ++k) {
    std::string key = r.read_string("key");
    ValueType t = read_value_type(r, key);
    Value v = read_value(r, t, key);
    if (!f.kv_index_.emplace(key, f.metadata_.size()).second) throw Error("gguf: duplicate key '" + escape_control(key) + "'");
    f.metadata_.emplace_back(std::move(key), std::move(v));
  }

  uint64_t alignment = kDefaultAlignment;
  if (const Value* a = f.find("general.alignment")) {
    const int64_t* p = std::get_if<int64_t>(a);
    if (!p) throw Error(std::string("gguf: general.alignment must be an integer, got ") + type_label(*a));
    // GGUF requires a multiple of 8; smaller values would misalign typed tensor loads.
    if (*p < 8 || (*p & (*p - 1)) != 0 || *p > (int64_t{1} << 30)) {
      throw Error("gguf: general.alignment " + std::to_string(*p) + " is not a power of two in [8, 2^30]");
    }
    alignment = static_cast<uint64_t>(*p);
  }

  f.tensors_.reserve(std::min<uint64_t>(n_tensors, 4096));
  for (uint64_t k = 0; k < n_tensors; ++k) {
    TensorInfo ti{};
    ti.name = r.read_string("tensor name");
    const std::string& name = ti.name;
    uint32_t n_dims = r.read<uint32_t>("tensor n_dims");
    if (n_dims < 1 || n_dims > 4) {
      throw Error("gguf: tensor '" + escape_control(name) + "' has n_dims " + std::to_string(n_dims) + " (must be 1..4)");
    }
    ti.tensor.n_dims = static_cast<int>(n_dims);
    for (uint32_t d = 0; d < n_dims; ++d) {
      uint64_t dim = r.read<uint64_t>("tensor dimension");
      if (dim == 0 || dim > kMaxI64) {
        throw Error("gguf: tensor '" + escape_control(name) + "' has invalid dimension " + std::to_string(dim));
      }
      ti.tensor.ne[d] = static_cast<int64_t>(dim);
    }
    uint32_t ggml_type = r.read<uint32_t>("tensor type");
    ti.offset = r.read<uint64_t>("tensor offset");

    switch (ggml_type) {
      case 0: case 1: case 2: case 8: ti.tensor.type = static_cast<DType>(ggml_type); break;
      default:
        throw Error("gguf: unsupported tensor type " + std::string(ggml_type_name(ggml_type)) + " (" +
                    std::to_string(ggml_type) + ") for " + escape_control(name) +
                    " \xe2\x80\x94 v1 supports F32/F16/Q8_0/Q4_0; convert with llama-quantize");
    }
    const DTypeInfo& info = dtype_info(ti.tensor.type);
    if (ti.tensor.ne[0] % info.block_size != 0) {
      throw Error("gguf: tensor '" + escape_control(name) + "' ne[0]=" + std::to_string(ti.tensor.ne[0]) + " is not a multiple of " +
                  info.name + " block size " + std::to_string(info.block_size));
    }
    uint64_t elems = 1;
    for (int d = 0; d < 4; ++d) {
      if (__builtin_mul_overflow(elems, static_cast<uint64_t>(ti.tensor.ne[d]), &elems) || elems > kMaxI64) {
        throw Error("gguf: tensor '" + escape_control(name) + "' element count overflows");
      }
    }
    uint64_t row_bytes;
    if (__builtin_mul_overflow(static_cast<uint64_t>(ti.tensor.ne[0] / info.block_size),
                               static_cast<uint64_t>(info.block_bytes), &row_bytes) ||
        row_bytes > kMaxI64) {
      throw Error("gguf: tensor '" + escape_control(name) + "' row byte size overflows");
    }
    uint64_t rows = static_cast<uint64_t>(ti.tensor.ne[1]) * static_cast<uint64_t>(ti.tensor.ne[2]) *
                    static_cast<uint64_t>(ti.tensor.ne[3]);  // <= elems, no overflow
    if (__builtin_mul_overflow(row_bytes, rows, &ti.nbytes) || ti.nbytes > kMaxI64) {
      throw Error("gguf: tensor '" + escape_control(name) + "' byte size overflows");
    }
    if (ti.offset % alignment != 0) {
      throw Error("gguf: tensor '" + escape_control(name) + "' offset " + std::to_string(ti.offset) + " is not a multiple of alignment " +
                  std::to_string(alignment));
    }
    if (!f.tensor_index_.emplace(name, f.tensors_.size()).second) {
      throw Error("gguf: duplicate tensor '" + escape_control(name) + "'");
    }
    f.tensors_.push_back(std::move(ti));
  }

  uint64_t data_start = align_up(r.pos(), alignment, "data section");
  for (TensorInfo& ti : f.tensors_) {
    uint64_t end;
    uint64_t avail = data_start <= bytes.size() ? bytes.size() - data_start : 0;
    if (__builtin_add_overflow(ti.offset, ti.nbytes, &end) || end > avail) {
      throw Error("gguf: tensor '" + escape_control(ti.name) + "' (offset " + std::to_string(ti.offset) + ", " +
                  std::to_string(ti.nbytes) + " bytes) extends past end of file");
    }
    ti.tensor.data = bytes.data() + data_start + ti.offset;
  }
  return f;
}

const Value* File::find(std::string_view key) const {
  auto it = kv_index_.find(key);
  return it == kv_index_.end() ? nullptr : &metadata_[it->second].second;
}

namespace {
const Value& require(const File& f, std::string_view key) {
  const Value* v = f.find(key);
  if (!v) throw Error("gguf: missing metadata key '" + escape_control(key) + "'");
  return *v;
}
[[noreturn]] void wrong_type(std::string_view key, const char* want, const Value& v) {
  throw Error("gguf: key '" + escape_control(key) + "' expected " + want + ", got " + type_label(v));
}
}  // namespace

std::string File::get_string(std::string_view key) const {
  const Value& v = require(*this, key);
  if (auto* p = std::get_if<std::string>(&v)) return *p;
  wrong_type(key, "string", v);
}

int64_t File::get_int(std::string_view key) const {
  const Value& v = require(*this, key);
  if (auto* p = std::get_if<int64_t>(&v)) return *p;
  wrong_type(key, "integer", v);
}

double File::get_float(std::string_view key) const {
  const Value& v = require(*this, key);
  if (auto* p = std::get_if<double>(&v)) return *p;
  wrong_type(key, "float", v);
}

const Array& File::get_array(std::string_view key) const {
  const Value& v = require(*this, key);
  if (auto* p = std::get_if<Array>(&v)) return *p;
  wrong_type(key, "array", v);
}

namespace {
[[noreturn]] void bad_access(const char* want, ValueType have, size_t i, size_t n) {
  throw Error(std::string("gguf: array access ") + want + "_at(" + std::to_string(i) + ") on array of type " +
              std::to_string(static_cast<uint32_t>(have)) + " with " + std::to_string(n) + " elements");
}
template <class T>
T load(const uint8_t* p) {
  T v;
  std::memcpy(&v, p, sizeof v);  // little-endian host assumed; no alignment requirement
  return v;
}
}  // namespace

int64_t Array::int_at(size_t i) const {
  if (!is_int() || i >= count_) bad_access("int", elem_type_, i, count_);
  const uint8_t* p = data_.data();
  switch (elem_type_) {
    case ValueType::U8: return load<uint8_t>(p + i);
    case ValueType::I8: return load<int8_t>(p + i);
    case ValueType::Bool: return load<uint8_t>(p + i) != 0;
    case ValueType::U16: return load<uint16_t>(p + 2 * i);
    case ValueType::I16: return load<int16_t>(p + 2 * i);
    case ValueType::U32: return load<uint32_t>(p + 4 * i);
    case ValueType::I32: return load<int32_t>(p + 4 * i);
    case ValueType::U64: return static_cast<int64_t>(load<uint64_t>(p + 8 * i));  // range-checked at parse
    case ValueType::I64: return load<int64_t>(p + 8 * i);
    default: bad_access("int", elem_type_, i, count_);
  }
}

double Array::float_at(size_t i) const {
  if (!is_float() || i >= count_) bad_access("float", elem_type_, i, count_);
  if (elem_type_ == ValueType::F32) return load<float>(data_.data() + 4 * i);
  return load<double>(data_.data() + 8 * i);
}

std::string_view Array::str_at(size_t i) const {
  if (!is_string() || i >= count_) bad_access("str", elem_type_, i, count_);
  const uint8_t* p = data_.data() + str_off_[i];
  const uint64_t len = load<uint64_t>(p);  // bounds validated at parse
  return {reinterpret_cast<const char*>(p + 8), static_cast<size_t>(len)};
}

const TensorInfo* File::find_tensor(std::string_view name) const {
  auto it = tensor_index_.find(name);
  return it == tensor_index_.end() ? nullptr : &tensors_[it->second];
}

}  // namespace nibble::gguf
