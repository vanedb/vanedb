// VaneDB - Copyright (c) 2025 Anton Tsvetkov - MIT License
#pragma once

#include <string>

#if defined(_WIN32) || defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace vanedb {
namespace detail {

/// Reopen a file by path, flush it to persistent media, close.
///
/// On macOS `fsync(2)` only pushes to the drive's write cache and returns, so
/// it does not guarantee the data reached the media. `F_FULLFSYNC` issues the
/// full barrier, which is what Rust's `File::sync_all()` does — the two
/// engines' save paths are only comparable if both wait for the same
/// guarantee (vanedb#110).
///
/// Silently no-ops if the file cannot be opened.
///
/// Caller must close any other writer (e.g. std::ofstream) for the same path
/// before calling — Windows CreateFileA fails on an exclusively-held file.
inline void fsync_file(const std::string& path) noexcept {
#if defined(_WIN32) || defined(_WIN64)
  HANDLE hFile = CreateFileA(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, NULL,
                             OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  if (hFile != INVALID_HANDLE_VALUE) { FlushFileBuffers(hFile); CloseHandle(hFile); }
#elif defined(__unix__) || defined(__APPLE__)
  int fd = open(path.c_str(), O_WRONLY);
  if (fd >= 0) {
#if defined(F_FULLFSYNC)
    // Some filesystems refuse F_FULLFSYNC (ENOTSUP); fsync is the fallback.
    if (fcntl(fd, F_FULLFSYNC) == -1) { fsync(fd); }
#else
    fsync(fd);
#endif
    close(fd);
  }
#endif
}

} // namespace detail
} // namespace vanedb
