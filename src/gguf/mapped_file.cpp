// POSIX implementation of MappedFile (open + fstat + mmap).
#include "gguf/mapped_file.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <string>
#include <system_error>
#include <utility>

#include "nibble/error.h"

namespace nibble::gguf {

namespace {
// The system's message for an errno value; std::error_code::message is thread-safe, unlike strerror.
std::string errno_message(int err) { return std::error_code(err, std::generic_category()).message(); }
}  // namespace

// The descriptor is closed as soon as the mapping exists (the mapping keeps the file alive). Every failure
// path closes it first so no descriptor leaks. MAP_PRIVATE + PROT_READ: the file is never modified.
MappedFile MappedFile::open(const std::string& path) {
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) throw Error("cannot open '" + path + "': " + errno_message(errno));
  struct stat st;
  if (::fstat(fd, &st) != 0) {
    std::string msg = "cannot stat '" + path + "': " + errno_message(errno);
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
  if (p == MAP_FAILED) throw Error("cannot mmap '" + path + "': " + errno_message(err));
  MappedFile f;
  f.addr_ = p;
  f.size_ = static_cast<size_t>(st.st_size);
  return f;
}

// Moves transfer ownership of the mapping; the source is left empty (it must not munmap).
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
