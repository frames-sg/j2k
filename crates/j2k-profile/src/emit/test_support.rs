// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{cell::RefCell, fmt, marker::PhantomData, rc::Rc};

thread_local! {
    static TEST_PROFILE_LINES: RefCell<Option<Vec<String>>> = const { RefCell::new(None) };
}

pub(super) fn capture(arguments: fmt::Arguments<'_>) -> bool {
    if !TEST_PROFILE_LINES.with(|slot| slot.borrow().is_some()) {
        return false;
    }
    let line = format!("{arguments}");
    TEST_PROFILE_LINES.with(|slot| {
        slot.borrow_mut()
            .as_mut()
            .expect("profile sink remained installed while formatting")
            .push(line);
    });
    true
}

pub(super) fn captured_profile_lines() -> Vec<String> {
    TEST_PROFILE_LINES.with(|slot| slot.borrow().clone().unwrap_or_default())
}

pub(super) struct TestProfileSinkGuard {
    previous: Option<Vec<String>>,
    _thread_bound: PhantomData<Rc<()>>,
}

pub(super) fn use_test_profile_sink() -> TestProfileSinkGuard {
    let previous = TEST_PROFILE_LINES.with(|slot| slot.replace(Some(Vec::new())));
    TestProfileSinkGuard {
        previous,
        _thread_bound: PhantomData,
    }
}

impl Drop for TestProfileSinkGuard {
    fn drop(&mut self) {
        TEST_PROFILE_LINES.with(|slot| {
            slot.replace(self.previous.take());
        });
    }
}
