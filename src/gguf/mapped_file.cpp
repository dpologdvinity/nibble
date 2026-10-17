#include "gguf/mapped_file.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <utility>

#include "nibble/error.h"

namespace nibble::gguf {

MappedFile MappedFile::open(const std::string& path) {
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) throw Error("cannot open '" + path + "': " + std::strerror(errno));
  struct stat st;
  if (::fstat(fd, &st) != 0) {
    std::string msg = "cannot stat '" + path + "': " + std::strerror(errno);
    ::close(fd);
    throw Error(msg);
  }
  if (!S_ISREG(st.st_mode) || st.st_size <= 0) {
    ::close(fd);
    throw Error("'" + path + "' is empty or not a regular file");
  }
  void* p = ::mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
  int err = errno;
  ::close(fd);
  if (p == MAP_FAILED) throw Error("cannot mmap '" + path + "': " + std::strerror(err));
  MappedFile f;
  f.addr_ = p;
  f.size_ = static_cast<size_t>(st.st_size);
  return f;
}

MappedFile::MappedFile(MappedFile&& o) noexcept
    : addr_(std::exchange(o.addr_, nullptr)), size_(std::exchange(o.size_, 0)) {}

MappedFile& MappedFile::operator=(MappedFile&& o) noexcept {
  if (this != &o) {
    reset();
    addr_ = std::exchange(o.addr_, nullptr);
    size_ = std::exchange(o.size_, 0);
  }
  return *this;
}

MappedFile::~MappedFile() { reset(); }

void MappedFile::reset() noexcept {
  if (addr_) ::munmap(addr_, size_);
  addr_ = nullptr;
  size_ = 0;
}

}  // namespace nibble::gguf
