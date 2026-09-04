//! The error type returned by every fallible operation.

/// Everything that can go wrong in this crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VaneError {
    /// A vector's length did not match the dimension the store was created with.
    DimensionMismatch {
        /// The dimension the store expects.
        expected: usize,
        /// The length actually supplied.
        got: usize,
    },
    /// A zero-length vector was supplied.
    EmptyVector,
    /// No vector is stored under this id.
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
    /// The index has reached the capacity it was built with.
    IndexFull,
    /// An input held a NaN or an infinity, which have no meaningful distance.
    NonFiniteValue {
        /// Which input was rejected, for the message.
        input: &'static str,
    },
    /// A parameter was outside its valid range, or an allocation it implies
    /// would overflow.
    InvalidParameter(&'static str),
    /// A filesystem or serialisation failure, with the underlying message.
    Io(String),
}

impl std::fmt::Display for VaneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::EmptyVector => write!(f, "empty vector"),
            Self::NotFound { id } => write!(f, "vector not found: {id}"),
            Self::DuplicateId { id } => write!(f, "duplicate id: {id}"),
            Self::InvalidK => write!(f, "k must be > 0"),
            Self::IndexFull => write!(f, "index is full"),
            Self::NonFiniteValue { input } => {
                write!(f, "{input} must contain only finite values")
            }
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::Io(msg) => write!(f, "I/O error: {msg}"),
        }
    }
}

impl std::error::Error for VaneError {}

/// `Result` with this crate's error type.
pub type Result<T> = std::result::Result<T, VaneError>;

#[cfg(test)]
mod tests {
    use super::*;

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
    fn error_display_index_full() {
        assert_eq!(VaneError::IndexFull.to_string(), "index is full");
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
}
