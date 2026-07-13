// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{cell::RefCell, ffi::OsString, marker::PhantomData, rc::Rc};

thread_local! {
    static TEST_NVIDIA_SMI_PROGRAM: RefCell<Option<OsString>> = const { RefCell::new(None) };
}

pub(super) fn nvidia_smi_program() -> OsString {
    TEST_NVIDIA_SMI_PROGRAM
        .with(|program| program.borrow().clone())
        .unwrap_or_else(|| OsString::from("nvidia-smi"))
}

pub(super) struct TestNvidiaSmiProgramGuard {
    previous: Option<OsString>,
    _thread_bound: PhantomData<Rc<()>>,
}

pub(super) fn use_test_nvidia_smi_program(program: OsString) -> TestNvidiaSmiProgramGuard {
    let previous = TEST_NVIDIA_SMI_PROGRAM.with(|slot| slot.replace(Some(program)));
    TestNvidiaSmiProgramGuard {
        previous,
        _thread_bound: PhantomData,
    }
}

impl Drop for TestNvidiaSmiProgramGuard {
    fn drop(&mut self) {
        TEST_NVIDIA_SMI_PROGRAM.with(|slot| {
            slot.replace(self.previous.take());
        });
    }
}
