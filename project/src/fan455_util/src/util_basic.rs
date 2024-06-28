#[macro_export]
macro_rules! elem {
    ($x:tt) => {
        ($x)
    };
    ($x:tt, $y:tt) => {
        ($x, $y)
    };
    ($x:tt, $y:tt, $($rest:tt),+) => {
        ($x, elem!($y, $($rest),+))
    };
}

#[macro_export]
macro_rules! mzip {
    ($x:expr) => {
        ($x)
    };
    ($x:expr, $y:expr) => {
        std::iter::zip($x, $y)
    };
    ($x:expr, $y:expr, $($rest:expr),+) => {
        std::iter::zip($x, mzip!($y, $($rest),+))
    };
}

#[macro_export]
macro_rules! assert_multi_eq {
    ($first:expr, $($other:expr),+) => {
        $(
            assert_eq!($first, $other);
        )+
    };
}

#[macro_export]
macro_rules! is_ascend {
    ($x:expr) => (true);
    ($x0:expr, $x1:expr) => (
        $x0 < $x1
    );
    ($x0:expr, $x1:expr $(, $xn:expr)*) => (
        $x0 < $x1 && is_ascend!($x1 $(, $xn)*)
    );
}

#[macro_export]
macro_rules! is_descend {
    ($x:expr) => (true);
    ($x0:expr, $x1:expr) => (
        $x0 > $x1
    );
    ($x0:expr, $x1:expr $(, $xn:expr)*) => (
        $x0 > $x1 && is_descend!($x1 $(, $xn)*)
    );
}
