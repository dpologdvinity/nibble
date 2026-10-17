#include <cstdint>
#include <cstdio>
#include <string>
#include <type_traits>
#include <variant>

#include "commands.h"
#include "gguf/gguf.h"
#include "nibble/error.h"
#include "tensor.h"

namespace nibble::cli {
namespace {

using gguf::ValueType;

const char* value_type_name(ValueType t) {
  switch (t) {
    case ValueType::U8: return "u8";
    case ValueType::I8: return "i8";
    case ValueType::U16: return "u16";
    case ValueType::I16: return "i16";
    case ValueType::U32: return "u32";
    case ValueType::I32: return "i32";
    case ValueType::F32: return "f32";
    case ValueType::Bool: return "bool";
    case ValueType::String: return "string";
    case ValueType::Array: return "array";
    case ValueType::U64: return "u64";
    case ValueType::I64: return "i64";
    case ValueType::F64: return "f64";
  }
  return "unknown";
}

std::string fmt_double(double d) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%g", d);
  return buf;
}

std::string fmt_array(const gguf::Array& a) {
  constexpr size_t kShow = 3;
  std::string out;
  for (size_t k = 0; k < a.size() && k < kShow; ++k) {
    if (k) out += ", ";
    if (a.is_string()) out += "\"" + std::string(a.str_at(k)) + "\"";
    else if (a.is_float()) out += fmt_double(a.float_at(k));
    else out += std::to_string(a.int_at(k));
  }
  return std::string("[") + value_type_name(a.elem_type()) + " x " + std::to_string(a.size()) + "] " + out + "...";
}

std::string fmt_value(const gguf::Value& v) {
  return std::visit(
      [](const auto& x) -> std::string {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<T, int64_t>) return std::to_string(x);
        else if constexpr (std::is_same_v<T, double>) return fmt_double(x);
        else if constexpr (std::is_same_v<T, bool>) return x ? "true" : "false";
        else if constexpr (std::is_same_v<T, std::string>) return x;
        else return fmt_array(x);
      },
      v);
}

}  // namespace

int cmd_info(int argc, char** argv) {
  if (argc != 2) throw Error("usage: nibble info <model.gguf>");
  const gguf::File f = gguf::File::open(argv[1]);

  std::printf("GGUF v%u\n\n", f.version());
  for (const auto& [key, val] : f.metadata()) std::printf("%s = %s\n", key.c_str(), fmt_value(val).c_str());

  std::printf("\ntensors: %zu\n", f.tensors().size());
  uint64_t total = 0;
  for (const auto& t : f.tensors()) {
    std::string dims;
    for (int d = 0; d < t.tensor.n_dims; ++d) {
      if (d) dims += ", ";
      dims += std::to_string(t.tensor.ne[static_cast<size_t>(d)]);
    }
    std::printf("%-40s %-6s [%s]  offset=%llu\n", t.name.c_str(),
                ggml_type_name(static_cast<uint32_t>(t.tensor.type)), dims.c_str(),
                static_cast<unsigned long long>(t.offset));
    total += t.nbytes;
  }
  std::printf("total tensor bytes: %llu (%.2f GiB)\n", static_cast<unsigned long long>(total),
              static_cast<double>(total) / (1024.0 * 1024.0 * 1024.0));
  return 0;
}

}  // namespace nibble::cli
