#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Example {
    pub stuff: String,
}

impl Example {
    pub fn new(value: String) -> Self {
        Example { stuff: value }
    }
}

/**********************************/
#[cfg(test)]
mod example_tests {
    use super::*;

    #[test]
    fn test_new() {
        let e = Example::new(String::from("test"));
        assert_eq!(e.stuff, String::from("test"));
    }

    #[test]
    fn test_clone_and_eq() {
        let e = Example::new(String::from("test"));
        assert_eq!(e, e.clone());
    }

    #[test]
    fn test_debug() {
        let e = Example::new(String::from("test"));
        assert_eq!(format!("{e:?}"), "Example { stuff: \"test\" }");
    }
}
