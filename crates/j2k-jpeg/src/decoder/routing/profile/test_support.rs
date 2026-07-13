// SPDX-License-Identifier: MIT OR Apache-2.0

use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use super::ProfileField;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct CapturedProfileRow {
    pub(super) operation: &'static str,
    pub(super) op: String,
    pub(super) path: String,
    pub(super) fields: Result<Vec<(String, String)>, String>,
}

thread_local! {
    static TEST_PROFILE_ROWS: RefCell<Option<Vec<CapturedProfileRow>>> =
        const { RefCell::new(None) };
}

pub(super) fn try_capture<const N: usize, F>(
    operation: &'static str,
    op: &str,
    path: &str,
    build: F,
) -> Result<(), F>
where
    F: FnOnce() -> j2k_profile::ProfileResult<[ProfileField; N]>,
{
    if !TEST_PROFILE_ROWS.with(|slot| slot.borrow().is_some()) {
        return Err(build);
    }

    let fields = build()
        .map(|fields| {
            fields
                .iter()
                .map(|field| (field.key().to_owned(), field.value().to_owned()))
                .collect()
        })
        .map_err(|error| error.to_string());
    TEST_PROFILE_ROWS.with(|slot| {
        slot.borrow_mut()
            .as_mut()
            .expect("profile sink remains installed while capturing")
            .push(CapturedProfileRow {
                operation,
                op: op.to_owned(),
                path: path.to_owned(),
                fields,
            });
    });
    Ok(())
}

pub(super) fn captured_profile_rows() -> Vec<CapturedProfileRow> {
    TEST_PROFILE_ROWS.with(|slot| slot.borrow().clone().unwrap_or_default())
}

pub(super) struct TestProfileSinkGuard {
    previous: Option<Vec<CapturedProfileRow>>,
    _thread_bound: PhantomData<Rc<()>>,
}

pub(super) fn use_test_profile_sink() -> TestProfileSinkGuard {
    let previous = TEST_PROFILE_ROWS.with(|slot| slot.replace(Some(Vec::new())));
    TestProfileSinkGuard {
        previous,
        _thread_bound: PhantomData,
    }
}

impl Drop for TestProfileSinkGuard {
    fn drop(&mut self) {
        TEST_PROFILE_ROWS.with(|slot| {
            slot.replace(self.previous.take());
        });
    }
}
