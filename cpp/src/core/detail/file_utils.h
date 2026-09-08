// VaneDB - Copyright (c) 2025 Anton Tsvetkov - MIT License
#pragma once

#include <atomic>
#include <cstdint>
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

/// A temporary path beside `dest`, unique per process and per writer.
///
/// A save writes here and renames into place, so a crash mid-write cannot
/// leave a half-written file. `dest + ".tmp"` is unique per destination —
/// unlike the extension-replacing form that collided in vanedb#38 — but two
/// concurrent saves to the *same* path still share one temp file and
/// interleave their writes, and the loser's rename publishes the corruption.
/// The pid and counter make the name unique per writer, matching what the
/// Rust engine does in `atomic_write.rs`.
///
/// Stays in the destination directory so the final rename is same-filesystem,
/// and therefore atomic.
inline std::string temp_path_for(const std::string& dest) {
  static std::atomic<uint64_t> sequence{0};
#if defined(_WIN32) || defined(_WIN64)
  const unsigned long pid = static_cast<unsigned long>(GetCurrentProcessId());
#else
  const unsigned long pid = static_cast<unsigned long>(getpid());
#endif
  return dest + "." + std::to_string(pid) + "." +
         std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)) + ".tmp";
}

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
