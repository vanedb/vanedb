//! The error type returned by every fallible operation.

use std::io;

/// Everything that can go wrong in this crate.
///
/// `#[non_exhaustive]`, so a `match` needs a `_` arm and a new variant is not
/// a breaking change. Not `PartialEq`, so message text stays out of the public
/// API — match on the variant. Not `Clone`, because [`io::Error`] is not.
#[derive(Debug)]
#[non_exhaustive]
pub enum VaneError {
    /// A vector's length did not match the dimension the store was created with.
    DimensionMismatch {
        /// The dimension the store expects.
        expected: usize,
        /// The length actually supplied.
        got: usize,
    },
    /// A batch's ids and vector data describe different row counts.
    BatchLengthMismatch {
        /// Ids supplied.
        ids: usize,
        /// Floats supplied; `ids * dim` were expected.
        vectors: usize,
        /// The index dimension.
        dim: usize,
    },
    /// A dimension of zero at construction. An empty vector passed to `add`
    /// reports [`VaneError::DimensionMismatch`] instead.
    ZeroDimension,
    /// No vector under this id. A lookup miss, not
    /// [`VaneError::FileNotFound`].
    NotFound {
        /// The id that was looked up.
        id: u64,
    },
    /// This id is already present; ids are unique within a store.
    DuplicateId {
        /// The id that was already taken.
        id: u64,
    },
    /// `k` was zero, so there is no nearest neighbour to return.
    InvalidK,
    /// An input held a NaN or an infinity, which have no meaningful distance.
    NonFiniteValue {
        /// Which input was rejected, for the message.
        input: &'static str,
    },
    /// A parameter was outside its valid range, or an allocation it implies
    /// would overflow.
    InvalidParameter(&'static str),
    /// The file does not exist, so "load it, or build it if it isn't there"
    /// can branch on the variant.
    FileNotFound {
        /// The operation that failed, such as `"open"`.
        context: &'static str,
        /// The underlying failure, whose [`io::Error::kind`] is
        /// [`io::ErrorKind::NotFound`].
        source: io::Error,
    },
    /// Readable, but not a valid vanedb structure. Retrying will not help.
    Corrupt {
        /// What was wrong with it, for diagnostics only. Not stable API.
        detail: String,
    },
    /// A compute backend (Metal, CUDA) is unavailable. Fall back to the CPU
    /// rather than retry.
    Backend {
        /// What failed, for diagnostics only. Not stable API.
        detail: String,
    },
    /// Any other filesystem or serialisation failure — a full disk, a
    /// permission problem, a failing device. Retrying may help.
    Io {
        /// The operation that failed, such as `"write"` or `"sync"`.
        context: &'static str,
        /// The underlying failure.
        source: io::Error,
    },
}

impl VaneError {
    /// Tags an [`io::Error`] with the operation that failed. A missing file
    /// becomes [`VaneError::FileNotFound`], everything else [`VaneError::Io`].
    pub fn from_io(context: &'static str, source: io::Error) -> Self {
        if source.kind() == io::ErrorKind::NotFound {
            Self::FileNotFound { context, source }
        } else {
            Self::Io { context, source }
        }
    }

    /// Reports a file whose contents are not a valid vanedb structure.
    pub fn corrupt(detail: impl Into<String>) -> Self {
        Self::Corrupt {
            detail: detail.into(),
        }
    }

    /// Reports a compute backend that could not be initialised or used.
    pub fn backend(detail: impl Into<String>) -> Self {
        Self::Backend {
            detail: detail.into(),
        }
    }
}

impl From<io::Error> for VaneError {
    fn from(source: io::Error) -> Self {
        Self::from_io("io", source)
    }
}

impl std::fmt::Display for VaneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::BatchLengthMismatch { ids, vectors, dim } => write!(
                f,
                "batch length mismatch: {ids} ids need {} floats at dimension {dim}, got {vectors}",
                ids.saturating_mul(*dim)
            ),
            Self::ZeroDimension => write!(f, "dimension must be > 0"),
            Self::NotFound { id } => write!(f, "vector not found: {id}"),
            Self::DuplicateId { id } => write!(f, "duplicate id: {id}"),
            Self::InvalidK => write!(f, "k must be > 0"),
            Self::NonFiniteValue { input } => {
                write!(f, "{input} must contain only finite values")
            }
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::FileNotFound { context, source } | Self::Io { context, source } => {
                write!(f, "{context}: {source}")
            }
            Self::Corrupt { detail } => write!(f, "corrupt file: {detail}"),
            Self::Backend { detail } => write!(f, "compute backend unavailable: {detail}"),
        }
    }
}

impl std::error::Error for VaneError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::FileNotFound { source, .. } | Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// `Result` with this crate's error type.
pub type Result<T> = std::result::Result<T, VaneError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    #[test]
    fn error_display_dimension_mismatch() {
        let err = VaneError::DimensionMismatch {
            expected: 768,
            got: 512,
        };
        assert_eq!(err.to_string(), "dimension mismatch: expected 768, got 512");
    }

    #[test]
    fn error_display_not_found() {
        let err = VaneError::NotFound { id: 42 };
        assert_eq!(err.to_string(), "vector not found: 42");
    }

    #[test]
    fn error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<VaneError>();
    }

    #[test]
    fn error_display_invalid_parameter() {
        let err = VaneError::InvalidParameter("M must be >= 2");
        assert_eq!(err.to_string(), "invalid parameter: M must be >= 2");
    }

    #[test]
    fn error_display_non_finite_value() {
        let err = VaneError::NonFiniteValue { input: "query" };
        assert_eq!(err.to_string(), "query must contain only finite values");
    }

    #[test]
    fn io_display_keeps_the_operation_that_failed() {
        let err = VaneError::from_io("write", io::Error::from(io::ErrorKind::StorageFull));
        assert!(err.to_string().starts_with("write: "), "{err}");
    }

    #[test]
    fn from_io_classifies_by_kind() {
        assert!(matches!(
            VaneError::from_io("open", io::Error::from(io::ErrorKind::NotFound)),
            VaneError::FileNotFound { .. }
        ));
        assert!(matches!(
            VaneError::from_io("open", io::Error::from(io::ErrorKind::PermissionDenied)),
            VaneError::Io { .. }
        ));
    }

    #[test]
    fn corrupt_carries_its_detail_and_no_source() {
        let err = VaneError::corrupt("invalid magic");
        assert_eq!(err.to_string(), "corrupt file: invalid magic");
        assert!(err.source().is_none());
    }
}
