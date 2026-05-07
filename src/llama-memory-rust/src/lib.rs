#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LlamaMemoryStatus {
    Success = 0,
    NoUpdate = 1,
    FailedPrepare = 2,
    FailedCompute = 3,
}

impl LlamaMemoryStatus {
    fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(Self::Success),
            1 => Some(Self::NoUpdate),
            2 => Some(Self::FailedPrepare),
            3 => Some(Self::FailedCompute),
            _ => None,
        }
    }
}

pub fn combine_statuses(s0: LlamaMemoryStatus, s1: LlamaMemoryStatus) -> LlamaMemoryStatus {
    let mut has_update = false;

    match s0 {
        LlamaMemoryStatus::Success => has_update = true,
        LlamaMemoryStatus::NoUpdate => {}
        LlamaMemoryStatus::FailedPrepare | LlamaMemoryStatus::FailedCompute => return s0,
    }

    match s1 {
        LlamaMemoryStatus::Success => has_update = true,
        LlamaMemoryStatus::NoUpdate => {}
        LlamaMemoryStatus::FailedPrepare | LlamaMemoryStatus::FailedCompute => return s1,
    }

    if has_update {
        LlamaMemoryStatus::Success
    } else {
        LlamaMemoryStatus::NoUpdate
    }
}

pub fn status_is_fail(status: LlamaMemoryStatus) -> bool {
    matches!(
        status,
        LlamaMemoryStatus::FailedPrepare | LlamaMemoryStatus::FailedCompute
    )
}

#[no_mangle]
pub extern "C" fn llama_memory_status_combine(s0: i32, s1: i32) -> i32 {
    let s0 = LlamaMemoryStatus::from_i32(s0).unwrap_or(LlamaMemoryStatus::FailedCompute);
    let s1 = LlamaMemoryStatus::from_i32(s1).unwrap_or(LlamaMemoryStatus::FailedCompute);
    combine_statuses(s0, s1) as i32
}

#[no_mangle]
pub extern "C" fn llama_memory_status_is_fail(status: i32) -> bool {
    LlamaMemoryStatus::from_i32(status).map_or(false, status_is_fail)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combines_no_update_only_as_no_update() {
        assert_eq!(
            combine_statuses(LlamaMemoryStatus::NoUpdate, LlamaMemoryStatus::NoUpdate),
            LlamaMemoryStatus::NoUpdate
        );
    }

    #[test]
    fn combines_any_success_as_success_without_failures() {
        assert_eq!(
            combine_statuses(LlamaMemoryStatus::Success, LlamaMemoryStatus::NoUpdate),
            LlamaMemoryStatus::Success
        );
        assert_eq!(
            combine_statuses(LlamaMemoryStatus::NoUpdate, LlamaMemoryStatus::Success),
            LlamaMemoryStatus::Success
        );
    }

    #[test]
    fn preserves_first_failure_by_order() {
        assert_eq!(
            combine_statuses(
                LlamaMemoryStatus::FailedPrepare,
                LlamaMemoryStatus::FailedCompute
            ),
            LlamaMemoryStatus::FailedPrepare
        );
        assert_eq!(
            combine_statuses(LlamaMemoryStatus::Success, LlamaMemoryStatus::FailedCompute),
            LlamaMemoryStatus::FailedCompute
        );
    }

    #[test]
    fn detects_failures() {
        assert!(!status_is_fail(LlamaMemoryStatus::Success));
        assert!(!status_is_fail(LlamaMemoryStatus::NoUpdate));
        assert!(status_is_fail(LlamaMemoryStatus::FailedPrepare));
        assert!(status_is_fail(LlamaMemoryStatus::FailedCompute));
    }
}
