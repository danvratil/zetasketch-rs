// SPDX-FileCopyrightText: 2025 Daniel Vrátil <me@dvratil.cz>
//
// SPDX-License-Identifier: MIT
//
// Based on the original Zetasketch implementation by Google:
// https://github.com/google/zetasketch
// Published under the Apache License 2.0

pub struct MergedIntIterator<I1: Iterator<Item = u32>, I2: Iterator<Item = u32>> {
    iter1: I1,
    iter2: I2,
    peek1: Option<u32>,
    peek2: Option<u32>,
}

impl<I1: Iterator<Item = u32>, I2: Iterator<Item = u32>> MergedIntIterator<I1, I2> {
    pub fn new(mut iter1: I1, mut iter2: I2) -> Self {
        let peek1 = iter1.next();
        let peek2 = iter2.next();
        Self {
            iter1,
            iter2,
            peek1,
            peek2,
        }
    }
}

impl<I1: Iterator<Item = u32>, I2: Iterator<Item = u32>> Iterator for MergedIntIterator<I1, I2> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        match (self.peek1, self.peek2) {
            (Some(v1), Some(v2)) => {
                if v1 <= v2 {
                    self.peek1 = self.iter1.next();
                    Some(v1)
                } else {
                    self.peek2 = self.iter2.next();
                    Some(v2)
                }
            }
            (Some(v1), None) => {
                self.peek1 = self.iter1.next();
                Some(v1)
            }
            (None, Some(v2)) => {
                self.peek2 = self.iter2.next();
                Some(v2)
            }
            (None, None) => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (i1, i1_bound) = self.iter1.size_hint();
        let (i2, i2_bound) = self.iter2.size_hint();

        (
            i1 + i2,
            match (i1_bound, i2_bound) {
                (Some(i1_bound), Some(i2_bound)) => Some(i1_bound + i2_bound),
                _ => None,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::MergedIntIterator;

    #[test]
    fn test_returns_values_in_sorted_order() {
        let data_a = vec![1, 2, 4];
        let data_b = vec![2, 3, 4, 5, 6, 7];

        assert_eq!(
            MergedIntIterator::new(data_a.into_iter(), data_b.into_iter()).collect::<Vec<_>>(),
            vec![1, 2, 2, 3, 4, 4, 5, 6, 7]
        );
    }
}
