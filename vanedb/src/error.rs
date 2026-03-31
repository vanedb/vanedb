#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VaneError {
    DimensionMismatch { expected: usize, got: usize },
    EmptyVector,
    NotFound { id: u64 },
    DuplicateId { id: u64 },
    InvalidK,
    IndexFull,
    InvalidParameter(&'static str),
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
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::Io(msg) => write!(f, "I/O error: {msg}"),
        }
    }
}

impl std::error::Error for VaneError {}

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
}
