//! Atomic file replacement.
//!
//! A save writes to a temporary sibling of its destination and renames it into
//! place, so a crash mid-write cannot leave a half-written file behind. The
//! temporary name must be unique per destination *and* per concurrent writer:
//! deriving it with `Path::with_extension("tmp")` made `index.bin` and
//! `index.idx` collide on `index.tmp`, so two unrelated saves truncated,
//! replaced, or renamed each other's work (#38).

use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::{Result, VaneError};

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// A temporary file that is removed when dropped, unless it was committed.
pub(crate) struct AtomicFile {
    temp: PathBuf,
    committed: bool,
}

impl AtomicFile {
    /// Reserve a temporary path beside `dest`. Staying in the destination
    /// directory keeps the final rename on one filesystem, and therefore
    /// atomic.
    pub(crate) fn new(dest: &Path) -> Self {
        let mut name = OsString::from(".");
        name.push(dest.file_name().unwrap_or_else(|| OsStr::new("vanedb")));
        name.push(format!(
            ".{}.{}.tmp",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        Self {
            temp: dest.with_file_name(name),
            committed: false,
        }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.temp
    }

    /// Publish the finished temporary file at `dest`.
    pub(crate) fn commit(mut self, dest: &Path) -> Result<()> {
        fs::rename(&self.temp, dest).map_err(|e| VaneError::Io(format!("rename: {e}")))?;
        self.committed = true;
        Ok(())
    }
}

impl Drop for AtomicFile {
    fn drop(&mut self) {
        if !self.committed {
            // Best effort: a failed save must not leave an orphan behind, but
            // there is nothing useful to do if the cleanup itself fails.
            let _ = fs::remove_file(&self.temp);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn destinations_sharing_a_stem_get_distinct_temp_paths() {
        let bin = AtomicFile::new(Path::new("/tmp/index.bin"));
        let idx = AtomicFile::new(Path::new("/tmp/index.idx"));
        assert_ne!(bin.path(), idx.path());
    }

    #[test]
    fn concurrent_saves_to_one_destination_get_distinct_temp_paths() {
        let first = AtomicFile::new(Path::new("/tmp/index.bin"));
        let second = AtomicFile::new(Path::new("/tmp/index.bin"));
        assert_ne!(first.path(), second.path());
    }

    #[test]
    fn temp_file_stays_in_the_destination_directory() {
        let dest = Path::new("/tmp/nested/index.bin");
        let temp = AtomicFile::new(dest);
        assert_eq!(temp.path().parent(), dest.parent());
    }
}
