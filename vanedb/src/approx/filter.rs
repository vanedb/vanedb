//! Filter definition and validation for restricted vector search.

use crate::error::{Result, VaneError};

/// Restriction on which external vector IDs can appear in search results.
///
/// Filters only govern result acceptance; they do not alter graph traversal,
/// ensuring graph connectivity is maintained even under highly selective filters.
///
/// # Examples
///
/// ```
/// use vanedb::approx::{Filter, SearchParams};
///
/// // Only accept ID 42 and ID 100
/// let allowed = [42, 100];
/// let filter = Filter::Allow(&allowed);
/// let params = SearchParams::new().filter(filter);
/// ```
#[derive(Clone, Copy)]
pub enum Filter<'a> {
    /// Accept an id when the predicate returns true.
    ///
    /// Predicates should be deterministic and may run more than once per ID
    /// during beam widening. They run while the index is locked for reading:
    /// do not access or modify that same index from the predicate. Querying a
    /// different index is supported.
    Predicate(&'a (dyn Fn(u64) -> bool + Sync)),
    /// Accept only these ids. Must be sorted in strictly ascending order with no duplicates.
    Allow(&'a [u64]),
    /// Accept every id except these. Must be sorted in strictly ascending order with no duplicates.
    Deny(&'a [u64]),
}

impl std::fmt::Debug for Filter<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Predicate(_) => f.debug_tuple("Predicate").finish(),
            Self::Allow(ids) => f.debug_tuple("Allow").field(ids).finish(),
            Self::Deny(ids) => f.debug_tuple("Deny").field(ids).finish(),
        }
    }
}

impl<'a> Filter<'a> {
    /// Checks whether the filter accepts an external ID.
    #[inline]
    pub fn accepts(&self, id: u64) -> bool {
        match self {
            Self::Predicate(pred) => pred(id),
            Self::Allow(ids) => ids.binary_search(&id).is_ok(),
            Self::Deny(ids) => ids.binary_search(&id).is_err(),
        }
    }

    /// Validates that `Allow` or `Deny` slices are sorted and deduplicated.
    pub(crate) fn validate(&self) -> Result<()> {
        match self {
            Self::Predicate(_) => Ok(()),
            Self::Allow(ids) | Self::Deny(ids) => {
                for window in ids.windows(2) {
                    if window[0] >= window[1] {
                        return Err(VaneError::Validation(
                            "filter id list must be sorted in strictly ascending order with no duplicates",
                        ));
                    }
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_debug() {
        let pred = |id: u64| id == 1;
        let f_pred = Filter::Predicate(&pred);
        assert_eq!(format!("{f_pred:?}"), "Predicate");

        let allow_slice = [10, 20];
        let f_allow = Filter::Allow(&allow_slice);
        assert_eq!(format!("{f_allow:?}"), "Allow([10, 20])");

        let deny_slice = [10, 20];
        let f_deny = Filter::Deny(&deny_slice);
        assert_eq!(format!("{f_deny:?}"), "Deny([10, 20])");
    }

    #[test]
    fn filter_accepts() {
        let pred = |id: u64| id % 2 == 0;
        let f_pred = Filter::Predicate(&pred);
        assert!(f_pred.validate().is_ok());
        assert!(f_pred.accepts(2));
        assert!(!f_pred.accepts(3));

        let allow_slice = [10, 20, 30];
        let f_allow = Filter::Allow(&allow_slice);
        assert!(f_allow.validate().is_ok());
        assert!(f_allow.accepts(10));
        assert!(!f_allow.accepts(15));

        let deny_slice = [10, 20, 30];
        let f_deny = Filter::Deny(&deny_slice);
        assert!(f_deny.validate().is_ok());
        assert!(!f_deny.accepts(10));
        assert!(f_deny.accepts(15));
    }

    #[test]
    fn filter_validation_rejects_unsorted_or_duplicates() {
        let unsorted = [20, 10, 30];
        assert!(Filter::Allow(&unsorted).validate().is_err());
        assert!(Filter::Deny(&unsorted).validate().is_err());

        let duplicates = [10, 20, 20, 30];
        assert!(Filter::Allow(&duplicates).validate().is_err());
        assert!(Filter::Deny(&duplicates).validate().is_err());
    }
}
