pub fn hello() -> &'static str {
    "vanedb"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(hello(), "vanedb");
    }
}
