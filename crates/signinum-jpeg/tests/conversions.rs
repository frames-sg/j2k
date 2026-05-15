use signinum_core::DecodeOutcome as CoreDecodeOutcome;
use signinum_jpeg::{DecodeOutcome, Rect, Warning};

#[test]
fn rect_conversions_preserve_coordinates_in_both_directions() {
    let jpeg = Rect {
        x: 3,
        y: 5,
        w: 7,
        h: 11,
    };

    let core: signinum_core::Rect = jpeg.into();
    assert_eq!(
        core,
        signinum_core::Rect {
            x: 3,
            y: 5,
            w: 7,
            h: 11,
        }
    );
    assert_eq!(Rect::from(core), jpeg);
}

#[test]
fn decode_outcome_conversion_preserves_rect_and_warnings() {
    let outcome = DecodeOutcome {
        decoded: Rect {
            x: 1,
            y: 2,
            w: 3,
            h: 4,
        },
        warnings: vec![Warning::MissingEoi],
    };

    let core: CoreDecodeOutcome<Warning> = outcome.into();
    assert_eq!(
        core.decoded,
        signinum_core::Rect {
            x: 1,
            y: 2,
            w: 3,
            h: 4,
        }
    );
    assert_eq!(core.warnings, vec![Warning::MissingEoi]);
}
