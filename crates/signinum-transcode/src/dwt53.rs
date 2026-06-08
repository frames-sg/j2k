// SPDX-License-Identifier: Apache-2.0

pub(crate) fn reversible_lift_53_i32(values: &mut [i32]) {
    let n = values.len();
    if n < 2 {
        return;
    }

    if n.is_multiple_of(2) {
        for i in (1..n - 1).step_by(2) {
            values[i] -= floor_div_i32(values[i - 1] + values[i + 1], 2);
        }
        values[n - 1] -= values[n - 2];

        values[0] += floor_div_i32(values[1] + 1, 2);
        for i in (2..n).step_by(2) {
            values[i] += floor_div_i32(values[i - 1] + values[i + 1] + 2, 4);
        }
        return;
    }

    let last_even = n - 1;
    for i in (1..n).step_by(2) {
        let right = values.get(i + 1).copied().unwrap_or(values[last_even]);
        values[i] -= floor_div_i32(values[i - 1] + right, 2);
    }
    for i in (0..n).step_by(2) {
        let left = if i > 0 { values[i - 1] } else { values[1] };
        let right = values.get(i + 1).copied().unwrap_or(left);
        values[i] += floor_div_i32(left + right + 2, 4);
    }
}

pub(crate) fn floor_div_i32(numerator: i32, denominator: i32) -> i32 {
    debug_assert!(denominator > 0);
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    if remainder != 0 && ((remainder < 0) != (denominator < 0)) {
        quotient - 1
    } else {
        quotient
    }
}

#[cfg(test)]
mod tests {
    use super::{floor_div_i32, reversible_lift_53_i32};

    #[test]
    fn floor_division_rounds_negative_values_down() {
        assert_eq!(floor_div_i32(-3, 2), -2);
        assert_eq!(floor_div_i32(3, 2), 1);
    }

    #[test]
    fn reversible_lift_matches_known_even_length_vector() {
        let mut values = [10, 20, 30, 40];

        reversible_lift_53_i32(&mut values);

        assert_eq!(values, [10, 0, 33, 10]);
    }

    #[test]
    fn reversible_lift_matches_known_odd_length_vector() {
        let mut values = [10, 20, 30, 40, 50];

        reversible_lift_53_i32(&mut values);

        assert_eq!(values, [10, 0, 30, 0, 50]);
    }
}
