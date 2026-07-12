//! Parsing a JP2 box, as specified in I.4.

use crate::error::{FormatError, Result};
use crate::reader::BitReader;

/// JP2 signature box - 'jP\040\040'.
pub(crate) const JP2_SIGNATURE: u32 = 0x6A50_2020;
/// File Type box - 'ftyp'.
pub(crate) const FILE_TYPE: u32 = 0x6674_7970;
/// JP2 Header box - 'jp2h'.
pub(crate) const JP2_HEADER: u32 = 0x6A70_3268;
/// Image Header box - 'ihdr'.
pub(crate) const IMAGE_HEADER: u32 = 0x6968_6472;
/// Bits Per Component box - 'bpcc'.
pub(crate) const BITS_PER_COMPONENT: u32 = 0x6270_6363;
/// Colour Specification box - 'colr'.
pub(crate) const COLOUR_SPECIFICATION: u32 = 0x636F_6C72;
/// Palette box - 'pclr'.
pub(crate) const PALETTE: u32 = 0x7063_6C72;
/// Component Mapping box - 'cmap'.
pub(crate) const COMPONENT_MAPPING: u32 = 0x636D_6170;
/// Channel Definition box - 'cdef'.
pub(crate) const CHANNEL_DEFINITION: u32 = 0x6364_6566;
/// Contiguous Codestream box - 'jp2c'.
pub(crate) const CONTIGUOUS_CODESTREAM: u32 = 0x6A70_3263;

pub(crate) struct Jp2Box<'a> {
    pub(crate) data: &'a [u8],
    pub(crate) box_type: u32,
}

#[cfg(test)]
pub(crate) fn read<'a>(reader: &mut BitReader<'a>) -> Option<Jp2Box<'a>> {
    read_checked(reader).ok()
}

pub(crate) fn read_checked<'a>(reader: &mut BitReader<'a>) -> Result<Jp2Box<'a>> {
    let offset = reader.offset();
    let l_box = reader
        .read_u32()
        .ok_or_else(|| box_header_truncated(reader, offset))?;
    let t_box = reader
        .read_u32()
        .ok_or_else(|| box_header_truncated(reader, offset))?;

    let data = match l_box {
        // If the value of this field is 0, then the length of the box
        // was not known when the LBox field was written. In this case, this box contains
        // all bytes up to the end of the file.
        0 => {
            let data = reader.tail().ok_or(FormatError::TruncatedAt {
                offset: reader.offset(),
                segment: "box payload",
            })?;
            reader.jump_to_end();
            data
        }
        // If the value of this field is 1, then the XLBox field shall exist and the value of
        // that field shall be the actual length of the box.
        // The value includes all of the fields of the box, including the LBox, TBox and XLBox
        // fields.
        1 => {
            let extended_offset = reader.offset();
            let xl_box = reader.read_u64().ok_or(FormatError::TruncatedAt {
                offset: extended_offset,
                segment: "extended box header",
            })?;
            let data_len = xl_box.checked_sub(16).ok_or(FormatError::InvalidBox)?;
            let data_len = usize::try_from(data_len).map_err(|_| FormatError::InvalidBox)?;
            read_box_payload(reader, data_len)?
        }
        // This field specifies the length of the box, stored as a 4-byte big-endian unsigned integer.
        // This value includes all of the fields of the box, including the length and type.
        _ => {
            let length = l_box.checked_sub(8).ok_or(FormatError::InvalidBox)?;
            let length = usize::try_from(length).map_err(|_| FormatError::InvalidBox)?;
            read_box_payload(reader, length)?
        }
    };

    Ok(Jp2Box {
        data,
        box_type: t_box,
    })
}

fn read_box_payload<'a>(reader: &mut BitReader<'a>, len: usize) -> Result<&'a [u8]> {
    let offset = reader.offset();
    let bytes = reader.read_bytes(len).ok_or(FormatError::TruncatedAt {
        offset,
        segment: "box payload",
    })?;
    Ok(bytes)
}

fn box_header_truncated(reader: &BitReader<'_>, offset: usize) -> FormatError {
    if offset == 0 && reader.offset() == 0 {
        FormatError::TooShort { need: 8, have: 0 }
    } else {
        FormatError::TruncatedAt {
            offset,
            segment: "box header",
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use crate::reader::BitReader;

    use super::{read, FILE_TYPE};

    #[test]
    fn read_extended_length_box() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u32.to_be_bytes());
        bytes.extend_from_slice(&FILE_TYPE.to_be_bytes());
        bytes.extend_from_slice(&18u64.to_be_bytes());
        bytes.extend_from_slice(b"jp");

        let mut reader = BitReader::new(&bytes);
        let parsed = read(&mut reader).expect("extended-length box parses");

        assert_eq!(parsed.box_type, FILE_TYPE);
        assert_eq!(parsed.data, b"jp");
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn read_rejects_extended_length_that_does_not_fit_usize() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u32.to_be_bytes());
        bytes.extend_from_slice(&FILE_TYPE.to_be_bytes());
        bytes.extend_from_slice(&(u64::from(u32::MAX) + 17).to_be_bytes());

        let mut reader = BitReader::new(&bytes);

        assert!(read(&mut reader).is_none());
    }
}
