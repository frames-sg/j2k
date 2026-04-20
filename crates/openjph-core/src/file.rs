//! File I/O abstractions — port of `ojph_file.h/cpp`.
//!
//! Provides trait-based input/output streams and concrete implementations
//! backed by files or in-memory buffers.

use std::fs::File;
use std::io::{self, Read, Write};

use crate::error::{OjphError, Result};

// ---------------------------------------------------------------------------
// SeekFrom
// ---------------------------------------------------------------------------

/// Origin for seek operations (mirrors [`std::io::SeekFrom`] but decoupled
/// from the standard library to match the C++ API).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeekFrom {
    /// Seek from the beginning of the stream.
    Start,
    /// Seek relative to the current position.
    Current,
    /// Seek from the end of the stream.
    End,
}

impl SeekFrom {
    /// Converts to the standard library equivalent, given an `offset`.
    fn to_std(self, offset: i64) -> io::SeekFrom {
        match self {
            Self::Start => io::SeekFrom::Start(offset as u64),
            Self::Current => io::SeekFrom::Current(offset),
            Self::End => io::SeekFrom::End(offset),
        }
    }
}

// ---------------------------------------------------------------------------
// Output trait
// ---------------------------------------------------------------------------

/// Trait for sequential / seekable output streams — port of `outfile_base`.
pub trait OutfileBase {
    /// Writes `data` and returns the number of bytes written.
    fn write(&mut self, data: &[u8]) -> Result<usize>;

    /// Returns the current byte position in the stream.
    fn tell(&self) -> i64 {
        0
    }

    /// Seeks to the given `offset` from `whence`.
    fn seek(&mut self, _offset: i64, _whence: SeekFrom) -> Result<()> {
        Err(OjphError::Unsupported("seek not supported".into()))
    }

    /// Flushes any buffered data.
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Input trait
// ---------------------------------------------------------------------------

/// Trait for sequential / seekable input streams — port of `infile_base`.
pub trait InfileBase {
    /// Reads up to `buf.len()` bytes, returning the count actually read.
    fn read(&mut self, buf: &mut [u8]) -> Result<usize>;

    /// Seeks to the given `offset` from `whence`.
    fn seek(&mut self, offset: i64, whence: SeekFrom) -> Result<()>;

    /// Returns the current byte position.
    fn tell(&self) -> i64;

    /// Returns `true` when the end of the stream has been reached.
    fn eof(&self) -> bool;
}

// =========================================================================
// J2cOutfile — wraps `std::fs::File`
// =========================================================================

/// File-backed output stream — port of `j2c_outfile`.
pub struct J2cOutfile {
    file: File,
    pos: i64,
}

impl J2cOutfile {
    /// Opens (creates / truncates) the file at `path`.
    pub fn open(path: &str) -> Result<Self> {
        let file = File::create(path)?;
        Ok(Self { file, pos: 0 })
    }
}

impl OutfileBase for J2cOutfile {
    fn write(&mut self, data: &[u8]) -> Result<usize> {
        let n = self.file.write(data)?;
        self.pos += n as i64;
        Ok(n)
    }

    fn tell(&self) -> i64 {
        self.pos
    }

    fn seek(&mut self, offset: i64, whence: SeekFrom) -> Result<()> {
        use std::io::Seek;
        let new_pos = self.file.seek(whence.to_std(offset))?;
        self.pos = new_pos as i64;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        self.file.flush()?;
        Ok(())
    }
}

// =========================================================================
// J2cInfile — wraps `std::fs::File`
// =========================================================================

/// File-backed input stream — port of `j2c_infile`.
pub struct J2cInfile {
    file: File,
    pos: i64,
    at_eof: bool,
}

impl J2cInfile {
    /// Opens an existing file for reading.
    pub fn open(path: &str) -> Result<Self> {
        let file = File::open(path)?;
        Ok(Self {
            file,
            pos: 0,
            at_eof: false,
        })
    }
}

impl InfileBase for J2cInfile {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        let n = self.file.read(buf)?;
        self.pos += n as i64;
        if n == 0 && !buf.is_empty() {
            self.at_eof = true;
        }
        Ok(n)
    }

    fn seek(&mut self, offset: i64, whence: SeekFrom) -> Result<()> {
        use std::io::Seek;
        let new_pos = self.file.seek(whence.to_std(offset))?;
        self.pos = new_pos as i64;
        self.at_eof = false;
        Ok(())
    }

    fn tell(&self) -> i64 {
        self.pos
    }

    fn eof(&self) -> bool {
        self.at_eof
    }
}

// =========================================================================
// MemOutfile — memory-backed output
// =========================================================================

/// In-memory output stream that grows as data is written — port of
/// `mem_outfile`.
pub struct MemOutfile {
    buf: Vec<u8>,
    pos: usize,
}

impl MemOutfile {
    /// Creates a new, empty memory output stream.
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            pos: 0,
        }
    }

    /// Creates a memory output stream with pre-allocated capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            buf: Vec::with_capacity(cap),
            pos: 0,
        }
    }

    /// Returns a reference to all data written so far.
    pub fn get_data(&self) -> &[u8] {
        &self.buf
    }

    /// Returns the total number of bytes written.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Returns `true` if no data has been written.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
}

impl Default for MemOutfile {
    fn default() -> Self {
        Self::new()
    }
}

impl OutfileBase for MemOutfile {
    fn write(&mut self, data: &[u8]) -> Result<usize> {
        if self.pos == self.buf.len() {
            self.buf.extend_from_slice(data);
        } else {
            // Overwrite existing bytes, extending if necessary.
            let end = self.pos + data.len();
            if end > self.buf.len() {
                self.buf.resize(end, 0);
            }
            self.buf[self.pos..end].copy_from_slice(data);
        }
        self.pos += data.len();
        Ok(data.len())
    }

    fn tell(&self) -> i64 {
        self.pos as i64
    }

    fn seek(&mut self, offset: i64, whence: SeekFrom) -> Result<()> {
        let new_pos = match whence {
            SeekFrom::Start => offset,
            SeekFrom::Current => self.pos as i64 + offset,
            SeekFrom::End => self.buf.len() as i64 + offset,
        };
        if new_pos < 0 {
            return Err(OjphError::InvalidParam("seek before start".into()));
        }
        self.pos = new_pos as usize;
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

// =========================================================================
// MemInfile — memory-backed input
// =========================================================================

/// Read-only input stream over a borrowed byte slice — port of `mem_infile`.
pub struct MemInfile<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> MemInfile<'a> {
    /// Creates a new memory input stream over `data`.
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
}

impl InfileBase for MemInfile<'_> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        let remaining = self.data.len().saturating_sub(self.pos);
        let n = buf.len().min(remaining);
        buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
        self.pos += n;
        Ok(n)
    }

    fn seek(&mut self, offset: i64, whence: SeekFrom) -> Result<()> {
        let new_pos = match whence {
            SeekFrom::Start => offset,
            SeekFrom::Current => self.pos as i64 + offset,
            SeekFrom::End => self.data.len() as i64 + offset,
        };
        if new_pos < 0 || new_pos as usize > self.data.len() {
            return Err(OjphError::InvalidParam("seek out of range".into()));
        }
        self.pos = new_pos as usize;
        Ok(())
    }

    fn tell(&self) -> i64 {
        self.pos as i64
    }

    fn eof(&self) -> bool {
        self.pos >= self.data.len()
    }
}
