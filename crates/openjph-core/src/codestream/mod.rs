//! JPEG 2000 codestream parser and generator.
//!
//! Port of `ojph_codestream.h/cpp`. The [`Codestream`] struct is the main
//! entry point for encoding and decoding HTJ2K images.
//!
//! # Encoding workflow
//!
//! 1. Create a [`Codestream`] and configure parameters via
//!    [`access_siz_mut()`](Codestream::access_siz_mut),
//!    [`access_cod_mut()`](Codestream::access_cod_mut), and
//!    [`access_qcd_mut()`](Codestream::access_qcd_mut).
//! 2. Call [`write_headers()`](Codestream::write_headers) to emit the
//!    main header to an [`OutfileBase`] implementor.
//! 3. Push image lines with [`exchange()`](Codestream::exchange).
//! 4. Call [`flush()`](Codestream::flush) to write tile data and the EOC marker.
//!
//! # Decoding workflow
//!
//! 1. Create a [`Codestream`] and call
//!    [`read_headers()`](Codestream::read_headers) on an [`InfileBase`] implementor.
//! 2. Optionally inspect SIZ/COD/QCD parameters.
//! 3. Call [`create()`](Codestream::create) to build internal structures.
//! 4. Pull decoded lines with [`pull()`](Codestream::pull).

pub mod bitbuffer_read;
pub mod bitbuffer_write;
pub(crate) mod codeblock;
pub(crate) mod codeblock_fun;
pub(crate) mod local;
pub(crate) mod precinct;
pub(crate) mod resolution;
pub(crate) mod subband;
pub(crate) mod tile;
pub(crate) mod tile_comp;

use crate::error::Result;
use crate::file::{InfileBase, OutfileBase};
use crate::params::{CommentExchange, ParamCod, ParamNlt, ParamQcd, ParamSiz};

/// The main codestream interface for encoding and decoding HTJ2K images.
///
/// `Codestream` is the public wrapper over the internal codec engine,
/// mirroring the C++ `ojph::codestream` class (which uses a pImpl pattern).
///
/// # Examples
///
/// **Lossless encode + decode round-trip (single-component 8-bit)**
///
/// ```rust
/// use openjph_core::codestream::Codestream;
/// use openjph_core::file::{MemOutfile, MemInfile};
/// use openjph_core::types::{Point, Size};
///
/// // Configure a minimal 8×8 grayscale image
/// let (w, h) = (8u32, 8u32);
/// let mut cs = Codestream::new();
/// cs.access_siz_mut().set_image_extent(Point::new(w, h));
/// cs.access_siz_mut().set_num_components(1);
/// cs.access_siz_mut().set_comp_info(0, Point::new(1, 1), 8, false);
/// cs.access_siz_mut().set_tile_size(Size::new(w, h));
/// cs.access_cod_mut().set_num_decomposition(0);
/// cs.access_cod_mut().set_reversible(true);
/// cs.access_cod_mut().set_color_transform(false);
/// cs.set_planar(0);
///
/// // Encode
/// let mut out = MemOutfile::new();
/// cs.write_headers(&mut out, &[]).unwrap();
/// let row = vec![42i32; w as usize];
/// for _ in 0..h { cs.exchange(&row, 0).unwrap(); }
/// cs.flush(&mut out).unwrap();
///
/// // Decode
/// let data = out.get_data().to_vec();
/// let mut inp = MemInfile::new(&data);
/// let mut dec = Codestream::new();
/// dec.read_headers(&mut inp).unwrap();
/// dec.create(&mut inp).unwrap();
/// for _ in 0..h {
///     let line = dec.pull(0).unwrap();
///     assert_eq!(line, row);
/// }
/// ```
#[derive(Debug, Default)]
pub struct Codestream {
    inner: local::CodestreamLocal,
}

impl Codestream {
    /// Creates a new codestream with default parameters.
    ///
    /// All marker segments (SIZ, COD, QCD, NLT) are initialised to sensible
    /// defaults. You must at least set the image extent, number of components,
    /// and component info before encoding.
    pub fn new() -> Self {
        Self {
            inner: local::CodestreamLocal::new(),
        }
    }

    /// Resets the codestream so it can be reused for another encode/decode
    /// operation without reallocating.
    pub fn restart(&mut self) {
        self.inner.restart();
    }

    // ----- Parameter access -----

    /// Returns a shared reference to the SIZ (image/tile size) parameters.
    pub fn access_siz(&self) -> &ParamSiz {
        self.inner.access_siz()
    }

    /// Returns a mutable reference to the SIZ parameters for configuration.
    ///
    /// Use this before calling [`write_headers()`](Self::write_headers) to set
    /// image extent, tile size, number of components, and per-component info.
    pub fn access_siz_mut(&mut self) -> &mut ParamSiz {
        self.inner.access_siz_mut()
    }

    /// Returns a shared reference to the COD (coding style) parameters.
    pub fn access_cod(&self) -> &ParamCod {
        self.inner.access_cod()
    }

    /// Returns a mutable reference to the COD parameters for configuration.
    ///
    /// Use this before calling [`write_headers()`](Self::write_headers) to set
    /// decomposition levels, reversibility, block sizes, and color transform.
    pub fn access_cod_mut(&mut self) -> &mut ParamCod {
        self.inner.access_cod_mut()
    }

    /// Returns a shared reference to the QCD (quantization) parameters.
    pub fn access_qcd(&self) -> &ParamQcd {
        self.inner.access_qcd()
    }

    /// Returns a mutable reference to the QCD parameters for configuration.
    ///
    /// For lossy (irreversible) compression, use
    /// [`set_delta()`](crate::params::ParamQcd::set_delta) to set the
    /// base quantization step size.
    pub fn access_qcd_mut(&mut self) -> &mut ParamQcd {
        self.inner.access_qcd_mut()
    }

    /// Returns a shared reference to the NLT (nonlinearity) parameters.
    pub fn access_nlt(&self) -> &ParamNlt {
        self.inner.access_nlt()
    }

    /// Returns a mutable reference to the NLT parameters for configuration.
    pub fn access_nlt_mut(&mut self) -> &mut ParamNlt {
        self.inner.access_nlt_mut()
    }

    // ----- Configuration -----

    /// Enables resilient (error-tolerant) mode for decoding.
    ///
    /// When enabled, the decoder attempts to recover from malformed tile-part
    /// headers rather than returning an error.
    pub fn enable_resilience(&mut self) {
        self.inner.enable_resilience();
    }

    /// Sets the line exchange mode.
    ///
    /// - `0` — interleaved (all components for each line exchanged together)
    /// - non-zero — planar (one component at a time)
    pub fn set_planar(&mut self, planar: i32) {
        self.inner.set_planar(planar);
    }

    /// Sets the codestream profile.
    ///
    /// Accepted values: `"IMF"`, `"BROADCAST"`, `"CINEMA2K"`, `"CINEMA4K"`, etc.
    ///
    /// # Errors
    ///
    /// Returns [`OjphError::InvalidParam`](crate::OjphError::InvalidParam) if
    /// `name` is not a recognised profile string.
    pub fn set_profile(&mut self, name: &str) -> Result<()> {
        self.inner.set_profile(name)
    }

    /// Sets tilepart division flags (bit-field).
    ///
    /// Use `0x1` for resolution-based divisions and `0x2` for
    /// component-based divisions.
    pub fn set_tilepart_divisions(&mut self, value: u32) {
        self.inner.set_tilepart_divisions(value);
    }

    /// Requests that TLM (tile-part length) markers be written in the
    /// codestream header.
    pub fn request_tlm_marker(&mut self, needed: bool) {
        self.inner.request_tlm_marker(needed);
    }

    #[cfg(test)]
    pub(crate) fn debug_inner(&self) -> &local::CodestreamLocal {
        &self.inner
    }

    /// Restricts the number of resolution levels used during decoding.
    ///
    /// `skipped_res_for_data` controls how many resolution levels are
    /// skipped when reading tile data. `skipped_res_for_recon` controls
    /// how many are skipped for reconstruction.
    pub fn restrict_input_resolution(
        &mut self,
        skipped_res_for_data: u32,
        skipped_res_for_recon: u32,
    ) {
        self.inner
            .restrict_input_resolution(skipped_res_for_data, skipped_res_for_recon);
    }

    // ----- Write path -----

    /// Writes the codestream main header (SOC through to the end of the
    /// main header) into `file`.
    ///
    /// Optional COM (comment) markers can be included via `comments`.
    ///
    /// # Errors
    ///
    /// Returns an error if parameter validation fails or writing to `file`
    /// encounters an I/O error.
    pub fn write_headers(
        &mut self,
        file: &mut dyn OutfileBase,
        comments: &[CommentExchange],
    ) -> Result<()> {
        self.inner.write_headers(file, comments)
    }

    /// Pushes one line of image data for the specified component.
    ///
    /// Call this repeatedly (height × num_components times in interleaved
    /// mode, or height times per component in planar mode) to supply the
    /// full image.
    ///
    /// Returns `Some(next_line_index)` while more lines are needed, or
    /// `None` when all lines for the current tile have been pushed.
    ///
    /// # Errors
    ///
    /// Returns an error if the internal encoder encounters a problem.
    pub fn exchange(&mut self, line: &[i32], comp_num: u32) -> Result<Option<usize>> {
        self.inner.exchange(line, comp_num)
    }

    /// Flushes the encoder: encodes all pending tiles and writes tile data
    /// plus the EOC (end of codestream) marker to `file`.
    ///
    /// # Errors
    ///
    /// Returns an error on I/O failure or if the encoder state is invalid.
    pub fn flush(&mut self, file: &mut dyn OutfileBase) -> Result<()> {
        self.inner.flush(file)
    }

    // ----- Read path -----

    /// Reads codestream main headers from `file`.
    ///
    /// After this call, you can inspect image parameters through
    /// [`access_siz()`](Self::access_siz) and friends.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream does not contain valid JPEG 2000
    /// codestream headers.
    pub fn read_headers(&mut self, file: &mut dyn InfileBase) -> Result<()> {
        self.inner.read_headers(file)
    }

    /// Builds internal decoding structures and decodes tile data from `file`.
    ///
    /// Must be called after [`read_headers()`](Self::read_headers). After
    /// this call, decoded lines can be retrieved with [`pull()`](Self::pull).
    ///
    /// # Errors
    ///
    /// Returns an error if decoding fails (corrupt data, unsupported
    /// features, etc.).
    pub fn create(&mut self, file: &mut dyn InfileBase) -> Result<()> {
        self.inner.create(file)
    }

    /// Pulls the next decoded line for the given `comp_num`.
    ///
    /// Returns `None` when all lines for this component have been returned.
    pub fn pull(&mut self, comp_num: u32) -> Option<Vec<i32>> {
        self.inner.pull(comp_num)
    }

    // ----- Query -----

    /// Returns `true` if planar mode is active.
    pub fn is_planar(&self) -> bool {
        self.inner.is_planar()
    }

    /// Returns the number of tiles in the x and y directions.
    pub fn get_num_tiles(&self) -> crate::types::Size {
        self.inner.num_tiles
    }
}
