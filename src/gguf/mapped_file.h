#pragma once
// RAII wrapper around a read-only mmap of a whole file: how model weights reach memory without being copied
// or read up front (the kernel pages them in on first touch, and clean pages can be dropped under pressure).
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace nibble::gguf {

// Read-only private mmap of a whole file. Move-only. The mapping address never changes for the lifetime of
// the object, even when the MappedFile itself is moved, so pointers into bytes() stay valid.
class MappedFile {
 public:
  // Maps the file at path. Throws nibble::Error if it cannot be opened, is empty or not a regular file, or
  // the mapping fails.
  static MappedFile open(const std::string& path);
  MappedFile() = default;  // an empty mapping: bytes() is an empty span
  MappedFile(MappedFile&& o) noexcept;
  MappedFile& operator=(MappedFile&& o) noexcept;
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  ~MappedFile();

  // The whole file as bytes. Valid until this object is destroyed or moved from.
  std::span<const uint8_t> bytes() const { return {static_cast<const uint8_t*>(addr_), size_}; }

 private:
  void reset() noexcept;  // munmap (if mapped) and return to the empty state
  void* addr_ = nullptr;
  size_t size_ = 0;
};

}  // namespace nibble::gguf
