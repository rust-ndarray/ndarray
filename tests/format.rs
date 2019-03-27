extern crate ndarray;

use ndarray::prelude::*;
use ndarray::{rcarr1, PRINT_ELEMENTS_LIMIT};

#[test]
fn formatting()
{
    let a = rcarr1::<f32>(&[1., 2., 3., 4.]);
    assert_eq!(format!("{}", a),
               //"[   1,    2,    3,    4]");
               "[1, 2, 3, 4]");
    assert_eq!(format!("{:4}", a),
               "[   1,    2,    3,    4]");
    let a = a.reshape((4, 1, 1));
    assert_eq!(format!("{:4}", a),
               "[[[   1]],\n [[   2]],\n [[   3]],\n [[   4]]]");

    let a = a.reshape((2, 2));
    assert_eq!(format!("{}", a),
               "[[1, 2],\n [3, 4]]");
    assert_eq!(format!("{}", a),
               "[[1, 2],\n [3, 4]]");
    assert_eq!(format!("{:4}", a),
               "[[   1,    2],\n [   3,    4]]");

    let b = arr0::<f32>(3.5);
    assert_eq!(format!("{}", b),
               "3.5");

    let s = format!("{:.3e}", aview1::<f32>(&[1.1, 2.2, 33., 440.]));
    assert_eq!(s,
               "[1.100e0, 2.200e0, 3.300e1, 4.400e2]");

    let s = format!("{:02x}", aview1::<u8>(&[1, 0xff, 0xfe]));
    assert_eq!(s, "[01, ff, fe]");
}

#[cfg(test)]
mod formatting_with_omit {
    use super::*;

    fn print_output_diff(expected: &str, actual: &str) {
        println!("Expected output:\n{}\nActual output:\n{}", expected, actual);
    }

    #[test]
    fn dim_1() {
        let overflow: usize = 5;
        let a = Array1::from_elem((PRINT_ELEMENTS_LIMIT * 2 + overflow, ), 1);
        let mut expected_output = String::from("[");
        a.iter()
            .take(PRINT_ELEMENTS_LIMIT)
            .for_each(|elem| { expected_output.push_str(format!("{}, ", elem).as_str()) });
        expected_output.push_str("...");
        a.iter()
            .skip(PRINT_ELEMENTS_LIMIT + overflow)
            .for_each(|elem| { expected_output.push_str(format!(", {}", elem).as_str()) });
        expected_output.push(']');
        let actual_output = format!("{}", a);

        print_output_diff(&expected_output, &actual_output);
        assert_eq!(actual_output, expected_output);
    }

    #[test]
    fn dim_2_last_axis_overflow() {
        let overflow: usize = 3;
        let a = Array2::from_elem((PRINT_ELEMENTS_LIMIT, PRINT_ELEMENTS_LIMIT * 2 + overflow), 1);
        let mut expected_output = String::from("[");

        for i in 0..PRINT_ELEMENTS_LIMIT {
            expected_output.push_str(format!("[{}", a[(i, 0)]).as_str());
            for j in 1..PRINT_ELEMENTS_LIMIT {
                expected_output.push_str(format!(", {}", a[(i, j)]).as_str());
            }
            expected_output.push_str(", ...");
            for j in PRINT_ELEMENTS_LIMIT + overflow..PRINT_ELEMENTS_LIMIT * 2 + overflow {
                expected_output.push_str(format!(", {}", a[(i, j)]).as_str());
            }
            expected_output.push_str(if i < PRINT_ELEMENTS_LIMIT - 1 { "],\n " } else { "]" });
        }
        expected_output.push(']');
        let actual_output = format!("{}", a);

        print_output_diff(&expected_output, &actual_output);
        assert_eq!(actual_output, expected_output);
    }

    #[test]
    fn dim_2_non_last_axis_overflow() {
        let overflow: usize = 5;
        let a = Array2::from_elem((PRINT_ELEMENTS_LIMIT * 2 + overflow, PRINT_ELEMENTS_LIMIT), 1);
        let mut expected_output = String::from("[");

        for i in 0..PRINT_ELEMENTS_LIMIT {
            expected_output.push_str(format!("[{}", a[(i, 0)]).as_str());
            for j in 1..PRINT_ELEMENTS_LIMIT {
                expected_output.push_str(format!(", {}", a[(i, j)]).as_str());
            }
            expected_output.push_str("],\n ");
        }
        expected_output.push_str("...,\n ");
        for i in PRINT_ELEMENTS_LIMIT + overflow..PRINT_ELEMENTS_LIMIT * 2 + overflow {
            expected_output.push_str(format!("[{}", a[(i, 0)]).as_str());
            for j in 1..PRINT_ELEMENTS_LIMIT {
                expected_output.push_str(format!(", {}", a[(i, j)]).as_str());
            }
            expected_output.push_str(if i == PRINT_ELEMENTS_LIMIT * 2 + overflow - 1 {
                "]"
            } else {
                "],\n "
            });
        }
        expected_output.push(']');
        let actual_output = format!("{}", a);

        print_output_diff(&expected_output, &actual_output);
        assert_eq!(actual_output, expected_output);
    }

    #[test]
    fn dim_2_multi_directional_overflow() {
        let overflow: usize = 5;
        let a = Array2::from_elem(
            (PRINT_ELEMENTS_LIMIT * 2 + overflow, PRINT_ELEMENTS_LIMIT * 2 + overflow), 1
        );
        let mut expected_output = String::from("[");

        for i in 0..PRINT_ELEMENTS_LIMIT {
            expected_output.push_str(format!("[{}", a[(i, 0)]).as_str());
            for j in 1..PRINT_ELEMENTS_LIMIT {
                expected_output.push_str(format!(", {}", a[(i, j)]).as_str());
            }
            expected_output.push_str(", ...");
            for j in PRINT_ELEMENTS_LIMIT + overflow..PRINT_ELEMENTS_LIMIT * 2 + overflow {
                expected_output.push_str(format!(", {}", a[(i, j)]).as_str());
            }
            expected_output.push_str("],\n ");
        }
        expected_output.push_str("...,\n ");
        for i in PRINT_ELEMENTS_LIMIT + overflow..PRINT_ELEMENTS_LIMIT * 2 + overflow {
            expected_output.push_str(format!("[{}", a[(i, 0)]).as_str());
            for j in 1..PRINT_ELEMENTS_LIMIT {
                expected_output.push_str(format!(", {}", a[(i, j)]).as_str());
            }
            expected_output.push_str(", ...");
            for j in PRINT_ELEMENTS_LIMIT + overflow..PRINT_ELEMENTS_LIMIT * 2 + overflow {
                expected_output.push_str(format!(", {}", a[(i, j)]).as_str());
            }
            expected_output.push_str(if i == PRINT_ELEMENTS_LIMIT * 2 + overflow - 1 {
                "]"
            } else {
                "],\n "
            });
        }
        expected_output.push(']');
        let actual_output = format!("{}", a);

        print_output_diff(&expected_output, &actual_output);
        assert_eq!(actual_output, expected_output);
    }
}

#[test]
fn debug_format() {
    let a = Array2::<i32>::zeros((3, 4));
    assert_eq!(
        format!("{:?}", a),
        "[[0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 0]] shape=[3, 4], strides=[4, 1], layout=C (0x1), const ndim=2"
    );
    assert_eq!(
        format!("{:?}", a.into_dyn()),
        "[[0, 0, 0, 0],
 [0, 0, 0, 0],
 [0, 0, 0, 0]] shape=[3, 4], strides=[4, 1], layout=C (0x1), dynamic ndim=2"
    );
}
