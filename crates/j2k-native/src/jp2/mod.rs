//! Reading a JP2 file, defined in Annex I.

mod allocation;
pub(crate) mod r#box;
pub(crate) mod cdef;
pub(crate) mod cmap;
pub(crate) mod colr;
mod container;
pub(crate) mod icc;
mod image_header;
mod metadata;
pub(crate) mod pclr;
mod validation;

pub use self::container::{
    extract_jp2_codestream_payload, inspect_jp2_container, Jp2Container, Jp2FileKind,
};
pub(crate) use self::container::{parse, parse_with_retained_baseline};
pub(crate) use self::metadata::{ComponentDescriptor, DecodedImage, ImageBoxes, ImageHeaderBox};
pub use self::metadata::{
    Jp2ChannelAssociation, Jp2ChannelDefinition, Jp2ChannelType, Jp2ColorSpec, Jp2ComponentMapping,
    Jp2ComponentMappingType, Jp2ComponentMetadata, Jp2FileMetadata, Jp2ImageHeaderMetadata,
    Jp2PaletteColumn, Jp2PaletteMetadata,
};

#[cfg(test)]
mod tests;
