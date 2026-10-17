#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace nibble::gguf {

// Read-only private mmap of a whole file. Move-only.
class MappedFile {
 public:
  static MappedFile open(const std::string& path);
  MappedFile() = default;
  MappedFile(MappedFile&& o) noexcept;
  MappedFile& operator=(MappedFile&& o) noexcept;
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  ~MappedFile();

  std::span<const uint8_t> bytes() const { return {static_cast<const uint8_t*>(addr_), size_}; }

 private:
  void reset() noexcept;
  void* addr_ = nullptr;
  size_t size_ = 0;
};

}  // namespace nibble::gguf
