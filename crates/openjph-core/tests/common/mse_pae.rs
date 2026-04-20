//! MSE (Mean Squared Error) and PAE (Peak Absolute Error) calculation.
//!
//! Port of `OpenJPH/tests/mse_pae.cpp`.
//! Computes MSE and PAE between two images stored as per-component sample arrays.

// =========================================================================
// Chroma subsampling formats
// =========================================================================

/// Image color subsampling format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ColorFormat {
    /// No subsampling — all components at full resolution.
    Format444,
    /// 4:2:2 horizontal subsampling for chroma.
    Format422,
    /// 4:2:0 both horizontal and vertical subsampling.
    Format420,
    /// Grayscale (single component).
    Format400,
}

// =========================================================================
// Image info
// =========================================================================

/// Downsampling factor for one component.
#[derive(Debug, Clone, Copy)]
pub struct Downsampling {
    pub x: u32,
    pub y: u32,
}

impl Default for Downsampling {
    fn default() -> Self {
        Self { x: 1, y: 1 }
    }
}

/// Image information and per-component sample data.
///
/// Stores decompressed image samples as `Vec<i32>` per component.
pub struct ImgInfo {
    pub num_comps: u32,
    pub width: usize,
    pub height: usize,
    pub downsampling: [Downsampling; 3],
    pub comps: Vec<Vec<i32>>,
    pub format: ColorFormat,
    pub bit_depth: u32,
    pub is_signed: bool,
}

impl ImgInfo {
    /// Create an empty image info.
    pub fn new() -> Self {
        Self {
            num_comps: 0,
            width: 0,
            height: 0,
            downsampling: [Downsampling::default(); 3],
            comps: Vec::new(),
            format: ColorFormat::Format444,
            bit_depth: 0,
            is_signed: false,
        }
    }

    /// Initialize image info and allocate component buffers.
    #[allow(dead_code)]
    pub fn init(
        &mut self,
        num_comps: u32,
        width: usize,
        height: usize,
        bit_depth: u32,
        is_signed: bool,
        format: ColorFormat,
    ) {
        assert!(num_comps <= 3);
        self.num_comps = num_comps;
        self.width = width;
        self.height = height;
        self.format = format;
        self.bit_depth = bit_depth;
        self.is_signed = is_signed;

        for i in 0..num_comps as usize {
            match format {
                ColorFormat::Format444 | ColorFormat::Format400 => {
                    self.downsampling[i] = Downsampling { x: 1, y: 1 };
                }
                ColorFormat::Format422 => {
                    self.downsampling[i] = Downsampling {
                        x: if i == 0 { 1 } else { 2 },
                        y: 1,
                    };
                }
                ColorFormat::Format420 => {
                    self.downsampling[i] = Downsampling {
                        x: if i == 0 { 1 } else { 2 },
                        y: if i == 0 { 1 } else { 2 },
                    };
                }
            }
        }

        self.comps = Vec::with_capacity(num_comps as usize);
        for i in 0..num_comps as usize {
            let w = (width as u32).div_ceil(self.downsampling[i].x);
            let h = (height as u32).div_ceil(self.downsampling[i].y);
            self.comps.push(vec![0i32; (w as usize) * (h as usize)]);
        }
    }

    /// Create an ImgInfo directly from per-component sample vectors.
    pub fn from_samples(
        width: usize,
        height: usize,
        bit_depth: u32,
        is_signed: bool,
        samples: Vec<Vec<i32>>,
    ) -> Self {
        let num_comps = samples.len() as u32;
        let mut img = Self::new();
        img.num_comps = num_comps;
        img.width = width;
        img.height = height;
        img.bit_depth = bit_depth;
        img.is_signed = is_signed;
        img.format = if num_comps == 1 {
            ColorFormat::Format400
        } else {
            ColorFormat::Format444
        };
        for i in 0..num_comps as usize {
            img.downsampling[i] = Downsampling { x: 1, y: 1 };
        }
        img.comps = samples;
        img
    }

    /// Component width accounting for downsampling.
    #[allow(dead_code)]
    pub fn comp_width(&self, comp: u32) -> usize {
        let ds = self.downsampling[comp as usize].x;
        (self.width as u32).div_ceil(ds) as usize
    }

    /// Component height accounting for downsampling.
    #[allow(dead_code)]
    pub fn comp_height(&self, comp: u32) -> usize {
        let ds = self.downsampling[comp as usize].y;
        (self.height as u32).div_ceil(ds) as usize
    }
}

impl Default for ImgInfo {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// MSE / PAE results
// =========================================================================

/// MSE and PAE results for one component.
#[derive(Debug, Clone, Copy)]
pub struct MsePaeResult {
    /// Mean Squared Error.
    pub mse: f32,
    /// Peak Absolute Error.
    pub pae: u32,
}

/// Compute MSE and PAE between two images.
///
/// Returns one `MsePaeResult` per component.
///
/// # Panics
///
/// Panics if the images do not match in dimensions, format, bit depth,
/// signedness, or number of components.
pub fn find_mse_pae(img1: &ImgInfo, img2: &ImgInfo) -> Vec<MsePaeResult> {
    assert_eq!(
        img1.num_comps, img2.num_comps,
        "mismatching number of components"
    );
    assert_eq!(img1.format, img2.format, "mismatching color formats");
    assert_eq!(img1.width, img2.width, "mismatching widths");
    assert_eq!(img1.height, img2.height, "mismatching heights");
    assert_eq!(img1.bit_depth, img2.bit_depth, "mismatching bit depths");
    assert_eq!(img1.is_signed, img2.is_signed, "mismatching signedness");

    let mut results = Vec::with_capacity(img1.num_comps as usize);

    for c in 0..img1.num_comps as usize {
        let w = (img1.width as u32).div_ceil(img1.downsampling[c].x);
        let h = (img1.height as u32).div_ceil(img1.downsampling[c].y);
        let w = w as usize;
        let h = h as usize;

        let mut se: f64 = 0.0;
        let mut lpae: u32 = 0;

        if img1.is_signed {
            for v in 0..h {
                for s in 0..w {
                    let idx = v * w + s;
                    let a = img1.comps[c][idx];
                    let b = img2.comps[c][idx];
                    let err = a - b;
                    let ae = err.unsigned_abs();
                    lpae = lpae.max(ae);
                    se += (err as f64) * (err as f64);
                }
            }
        } else {
            for v in 0..h {
                for s in 0..w {
                    let idx = v * w + s;
                    let a = img1.comps[c][idx] as u32;
                    let b = img2.comps[c][idx] as u32;
                    let err = a.abs_diff(b);
                    lpae = lpae.max(err);
                    se += (err as f64) * (err as f64);
                }
            }
        }

        let mse = (se / (w * h) as f64) as f32;
        results.push(MsePaeResult { mse, pae: lpae });
    }

    results
}

/// Compute MSE and PAE with NLT (Non-Linearity Transform) adjustment.
///
/// For signed components, applies the NLT bias: negative values are mapped as
/// `−|v| − (2^(B−1) + 1)` before comparison.
pub fn find_nlt_mse_pae(img1: &ImgInfo, img2: &ImgInfo) -> Vec<MsePaeResult> {
    assert_eq!(
        img1.num_comps, img2.num_comps,
        "mismatching number of components"
    );
    assert_eq!(img1.format, img2.format, "mismatching color formats");
    assert_eq!(img1.width, img2.width, "mismatching widths");
    assert_eq!(img1.height, img2.height, "mismatching heights");
    assert_eq!(img1.bit_depth, img2.bit_depth, "mismatching bit depths");
    assert_eq!(img1.is_signed, img2.is_signed, "mismatching signedness");

    let mut results = Vec::with_capacity(img1.num_comps as usize);

    for c in 0..img1.num_comps as usize {
        let w = (img1.width as u32).div_ceil(img1.downsampling[c].x);
        let h = (img1.height as u32).div_ceil(img1.downsampling[c].y);
        let w = w as usize;
        let h = h as usize;

        let mut se: f64 = 0.0;
        let mut lpae: u32 = 0;

        if img1.is_signed {
            let bias = (1i32 << (img1.bit_depth - 1)) + 1;
            for v in 0..h {
                for s in 0..w {
                    let idx = v * w + s;
                    let mut a = img1.comps[c][idx];
                    let mut b = img2.comps[c][idx];
                    a = if a >= 0 { a } else { -a - bias };
                    b = if b >= 0 { b } else { -b - bias };
                    let err = (a as u32).abs_diff(b as u32);
                    lpae = lpae.max(err);
                    se += (err as f64) * (err as f64);
                }
            }
        } else {
            for v in 0..h {
                for s in 0..w {
                    let idx = v * w + s;
                    let a = img1.comps[c][idx] as u32;
                    let b = img2.comps[c][idx] as u32;
                    let err = a.abs_diff(b);
                    lpae = lpae.max(err);
                    se += (err as f64) * (err as f64);
                }
            }
        }

        let mse = (se / (w * h) as f64) as f32;
        results.push(MsePaeResult { mse, pae: lpae });
    }

    results
}

// =========================================================================
// PGM/PPM file loading
// =========================================================================

/// Check if a filename is a PNM (PGM/PPM) file.
#[allow(dead_code)]
pub fn is_pnm(filename: &str) -> bool {
    let lower = filename.to_lowercase();
    lower.ends_with(".pgm") || lower.ends_with(".ppm")
}

/// Load a PGM (grayscale) or PPM (RGB) image file into `ImgInfo`.
#[allow(dead_code)]
pub fn load_ppm(filename: &str) -> std::io::Result<ImgInfo> {
    let data = std::fs::read(filename)?;
    let mut pos = 2;

    fn skip_whitespace_and_comments(data: &[u8], pos: &mut usize) {
        while *pos < data.len() {
            if data[*pos] == b'#' {
                // Skip comment line
                while *pos < data.len() && data[*pos] != b'\n' {
                    *pos += 1;
                }
                if *pos < data.len() {
                    *pos += 1;
                }
            } else if data[*pos].is_ascii_whitespace() {
                *pos += 1;
            } else {
                break;
            }
        }
    }

    fn read_number(data: &[u8], pos: &mut usize) -> u32 {
        skip_whitespace_and_comments(data, pos);
        let start = *pos;
        while *pos < data.len() && data[*pos].is_ascii_digit() {
            *pos += 1;
        }
        let s = std::str::from_utf8(&data[start..*pos]).unwrap_or("0");
        s.parse::<u32>().unwrap_or(0)
    }

    // Read magic number
    if data.len() < 2 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "file too short",
        ));
    }
    let magic = [data[0], data[1]];

    let num_comps = match &magic {
        b"P5" => 1, // PGM binary
        b"P6" => 3, // PPM binary
        _ => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unsupported PNM magic {:?}", magic),
            ));
        }
    };

    let width = read_number(&data, &mut pos) as usize;
    let height = read_number(&data, &mut pos) as usize;
    let max_val = read_number(&data, &mut pos);

    // Skip one whitespace character after maxval
    if pos < data.len() && data[pos].is_ascii_whitespace() {
        pos += 1;
    }

    let bit_depth = if max_val <= 255 { 8 } else { 16 };
    let bytes_per_sample = if bit_depth <= 8 { 1 } else { 2 };

    let mut img = ImgInfo::new();
    let format = if num_comps == 1 {
        ColorFormat::Format400
    } else {
        ColorFormat::Format444
    };
    img.init(num_comps, width, height, bit_depth, false, format);

    // Read pixel data
    if num_comps == 1 {
        // PGM: samples in raster order
        for y in 0..height {
            for x in 0..width {
                let val = if bytes_per_sample == 1 {
                    data[pos] as i32
                } else {
                    let v = u16::from_be_bytes([data[pos], data[pos + 1]]);
                    v as i32
                };
                pos += bytes_per_sample;
                img.comps[0][y * width + x] = val;
            }
        }
    } else {
        // PPM: R,G,B interleaved
        for y in 0..height {
            for x in 0..width {
                for c in 0..3 {
                    let val = if bytes_per_sample == 1 {
                        data[pos] as i32
                    } else {
                        let v = u16::from_be_bytes([data[pos], data[pos + 1]]);
                        v as i32
                    };
                    pos += bytes_per_sample;
                    img.comps[c][y * width + x] = val;
                }
            }
        }
    }

    Ok(img)
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mse_pae_identical_unsigned() {
        let samples = vec![vec![10, 20, 30, 40]];
        let img1 = ImgInfo::from_samples(2, 2, 8, false, samples.clone());
        let img2 = ImgInfo::from_samples(2, 2, 8, false, samples);
        let results = find_mse_pae(&img1, &img2);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].mse, 0.0);
        assert_eq!(results[0].pae, 0);
    }

    #[test]
    fn mse_pae_different_unsigned() {
        let img1 = ImgInfo::from_samples(2, 2, 8, false, vec![vec![10, 20, 30, 40]]);
        let img2 = ImgInfo::from_samples(2, 2, 8, false, vec![vec![12, 20, 28, 40]]);
        let results = find_mse_pae(&img1, &img2);
        assert_eq!(results.len(), 1);
        // Errors: 2, 0, 2, 0 → MSE = (4+0+4+0)/4 = 2.0, PAE = 2
        assert_eq!(results[0].mse, 2.0);
        assert_eq!(results[0].pae, 2);
    }

    #[test]
    fn mse_pae_signed() {
        let img1 = ImgInfo::from_samples(2, 2, 8, true, vec![vec![-10, 20, -30, 40]]);
        let img2 = ImgInfo::from_samples(2, 2, 8, true, vec![vec![-10, 20, -30, 40]]);
        let results = find_mse_pae(&img1, &img2);
        assert_eq!(results[0].mse, 0.0);
        assert_eq!(results[0].pae, 0);
    }

    #[test]
    fn mse_pae_multi_component() {
        let img1 = ImgInfo::from_samples(
            2,
            2,
            8,
            false,
            vec![
                vec![10, 20, 30, 40],
                vec![0, 0, 0, 0],
                vec![255, 255, 255, 255],
            ],
        );
        let img2 = ImgInfo::from_samples(
            2,
            2,
            8,
            false,
            vec![
                vec![10, 20, 30, 40],
                vec![1, 1, 1, 1],
                vec![254, 254, 254, 254],
            ],
        );
        let results = find_mse_pae(&img1, &img2);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].mse, 0.0);
        assert_eq!(results[0].pae, 0);
        assert_eq!(results[1].mse, 1.0);
        assert_eq!(results[1].pae, 1);
        assert_eq!(results[2].mse, 1.0);
        assert_eq!(results[2].pae, 1);
    }

    #[test]
    fn nlt_mse_pae_unsigned() {
        let img1 = ImgInfo::from_samples(2, 2, 8, false, vec![vec![10, 20, 30, 40]]);
        let img2 = ImgInfo::from_samples(2, 2, 8, false, vec![vec![10, 20, 30, 40]]);
        let results = find_nlt_mse_pae(&img1, &img2);
        assert_eq!(results[0].mse, 0.0);
        assert_eq!(results[0].pae, 0);
    }
}
