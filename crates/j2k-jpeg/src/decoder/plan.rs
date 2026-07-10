// SPDX-License-Identifier: MIT OR Apache-2.0

//! Prepared decode-plan construction and validation.

use super::{
    checked_usize_product, compute_decode_scratch_bytes, compute_extended12_planes_scratch_bytes,
    compute_lossless_scratch_bytes, lossless_color_sampling, Arc, ColorSpace, Decoder,
    DecoderContext, HuffmanTable, HuffmanValues, Info, JpegError, MarkerKind, ParsedHeader,
    PreparedComponentPlan, PreparedDecodePlan, PreparedLosslessPlan,
    PreparedProgressiveComponentPlan, PreparedProgressivePlan, PreparedProgressiveScan,
    PreparedProgressiveScanComponent, RawHuffmanTable, SofKind, Vec, DEFAULT_MAX_DECODE_BYTES,
};

impl Decoder<'_> {
    pub(super) fn build_prepared_plan(
        header: &ParsedHeader,
        info: &Info,
        ctx: &mut DecoderContext,
    ) -> Result<PreparedDecodePlan, JpegError> {
        match info.sof_kind {
            SofKind::Baseline8 | SofKind::Extended8 | SofKind::Extended12 => {}
            other => return Err(JpegError::NotImplemented { sof: other }),
        }
        if info.sof_kind == SofKind::Extended12
            && !matches!(
                info.color_space,
                ColorSpace::Grayscale
                    | ColorSpace::YCbCr
                    | ColorSpace::Rgb
                    | ColorSpace::Cmyk
                    | ColorSpace::Ycck
            )
        {
            return Err(JpegError::NotImplemented { sof: info.sof_kind });
        }
        match info.color_space {
            ColorSpace::Grayscale
            | ColorSpace::YCbCr
            | ColorSpace::Rgb
            | ColorSpace::Cmyk
            | ColorSpace::Ycck => {}
        }

        let mut dc_tables: [Option<Arc<HuffmanTable>>; 4] = Default::default();
        let mut ac_tables: [Option<Arc<HuffmanTable>>; 4] = Default::default();
        let scan = header.scan.as_ref().ok_or(JpegError::MissingMarker {
            marker: MarkerKind::Sos,
        })?;
        if header.scan_count != 1 {
            return Err(JpegError::InvalidSequentialScanCount {
                sof: info.sof_kind,
                count: header.scan_count,
            });
        }
        validate_leading_component_sampling(header, info)?;
        // Every component must declare H,V in 1..=4 per T.81 §B.2.2, and max_h
        // must actually divide every component's H (same for V). Malformed
        // streams can set H=0 (div-by-zero in upsample ratio), non-divisors
        // (arbitrary ratios M2 handles), or ratios that don't produce planes
        // that cover the image width.
        for (h, v) in header.sampling.iter() {
            if h == 0 || v == 0 || h > 4 || v > 4 {
                return Err(JpegError::NotImplemented { sof: info.sof_kind });
            }
            if !header.sampling.max_h.is_multiple_of(h) || !header.sampling.max_v.is_multiple_of(v)
            {
                return Err(JpegError::NotImplemented { sof: info.sof_kind });
            }
        }
        for comp in &scan.components {
            let di = comp.dc_table as usize;
            let ai = comp.ac_table as usize;
            if dc_tables[di].is_none() {
                let raw = header.huffman_tables.dc[di].as_ref().ok_or(
                    JpegError::MissingHuffmanTable {
                        component: comp.id,
                        class: 0,
                        id: comp.dc_table,
                    },
                )?;
                dc_tables[di] = Some(ctx.resolve_huffman_table(raw)?);
            }
            if ac_tables[ai].is_none() {
                let raw = header.huffman_tables.ac[ai].as_ref().ok_or(
                    JpegError::MissingHuffmanTable {
                        component: comp.id,
                        class: 1,
                        id: comp.ac_table,
                    },
                )?;
                ac_tables[ai] = Some(ctx.resolve_huffman_table(raw)?);
            }
        }

        build_decode_plan(header, info, &dc_tables, &ac_tables, ctx)
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "JPEG component counts are validated against the eight-bit SOF field"
    )]
    pub(super) fn build_lossless_plan(
        header: &ParsedHeader,
        info: &Info,
        ctx: &mut DecoderContext,
    ) -> Result<(PreparedDecodePlan, PreparedLosslessPlan), JpegError> {
        if info.sof_kind != SofKind::Lossless {
            return Err(JpegError::NotImplemented { sof: info.sof_kind });
        }
        if header.scan_count != 1 {
            return Err(JpegError::NotImplemented { sof: info.sof_kind });
        }
        let scan = header.scan.as_ref().ok_or(JpegError::MissingMarker {
            marker: MarkerKind::Sos,
        })?;
        if !(1..=7).contains(&scan.ss) {
            return Err(JpegError::UnsupportedPredictor { predictor: scan.ss });
        }
        if scan.se != 0 || scan.ah != 0 || scan.al != 0 {
            return Err(JpegError::NotImplemented { sof: info.sof_kind });
        }
        let expected_components = match (info.color_space, info.bit_depth) {
            (ColorSpace::Grayscale, 8 | 16) => 1,
            (ColorSpace::Rgb | ColorSpace::YCbCr, 8 | 16) => 3,
            _ => return Err(JpegError::NotImplemented { sof: info.sof_kind }),
        };
        if scan.components.len() != expected_components {
            return Err(JpegError::UnsupportedComponentCount {
                count: scan.components.len() as u8,
            });
        }
        let empty_raw = RawHuffmanTable {
            bits: [0; 16],
            values: HuffmanValues::default(),
        };
        let empty_huffman = ctx.resolve_huffman_table(&empty_raw)?;
        let mut components = Vec::with_capacity(scan.components.len());
        let mut first_dc_table = None;
        for scan_component in scan.components.iter().copied() {
            let component_index = find_component_index(&header.component_ids, scan_component.id)
                .ok_or(JpegError::UnknownScanComponent {
                    offset: header.sos_offset.unwrap_or_default(),
                    component: scan_component.id,
                })?;
            let (h, v) =
                header
                    .sampling
                    .component(component_index)
                    .ok_or(JpegError::MissingMarker {
                        marker: MarkerKind::Sof,
                    })?;
            let raw_dc = header.huffman_tables.dc[scan_component.dc_table as usize]
                .as_ref()
                .ok_or(JpegError::MissingHuffmanTable {
                    component: scan_component.id,
                    class: 0,
                    id: scan_component.dc_table,
                })?;
            let dc_table = ctx.resolve_huffman_table(raw_dc)?;
            first_dc_table.get_or_insert_with(|| Arc::clone(&dc_table));
            components.push(PreparedComponentPlan {
                h,
                v,
                output_index: component_index,
                quant: ctx.resolve_quant_table([1; 64]),
                dc_table,
                ac_table: Arc::clone(&empty_huffman),
            });
        }
        if matches!(info.color_space, ColorSpace::Rgb | ColorSpace::YCbCr)
            && lossless_color_sampling(info).is_none()
        {
            return Err(JpegError::NotImplemented { sof: info.sof_kind });
        }
        let plan = PreparedDecodePlan {
            components,
            sampling: info.sampling,
            color_space: info.color_space,
            restart_interval: header.restart_interval,
            dimensions: info.dimensions,
            scan_offset: header.sos_offset.ok_or(JpegError::MissingMarker {
                marker: MarkerKind::Sos,
            })?,
            scratch_bytes: compute_lossless_scratch_bytes(info, DEFAULT_MAX_DECODE_BYTES)?,
        };
        let lossless = PreparedLosslessPlan {
            predictor: scan.ss,
            bit_depth: info.bit_depth,
            dc_table: first_dc_table.ok_or(JpegError::MissingMarker {
                marker: MarkerKind::Sos,
            })?,
            dimensions: info.dimensions,
            scan_offset: plan.scan_offset,
        };
        Ok((plan, lossless))
    }

    #[expect(
        clippy::too_many_lines,
        reason = "progressive planning validates the ordered scan script while compiling shared table state"
    )]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "JPEG scan component counts are validated against the eight-bit SOS field"
    )]
    pub(super) fn build_progressive_plan(
        header: &ParsedHeader,
        info: &Info,
        ctx: &mut DecoderContext,
    ) -> Result<PreparedProgressivePlan, JpegError> {
        if !matches!(
            info.sof_kind,
            SofKind::Progressive8 | SofKind::Progressive12
        ) {
            return Err(JpegError::NotImplemented { sof: info.sof_kind });
        }
        match (info.sof_kind, info.color_space) {
            (
                SofKind::Progressive8 | SofKind::Progressive12,
                ColorSpace::Grayscale | ColorSpace::YCbCr | ColorSpace::Rgb,
            )
            | (SofKind::Progressive12, ColorSpace::Cmyk | ColorSpace::Ycck) => {}
            (_, color_space) => return Err(JpegError::UnsupportedColorSpace { color_space }),
        }
        validate_sampling_factors(header, info)?;
        if header.progressive_scans.is_empty() {
            return Err(JpegError::MissingMarker {
                marker: MarkerKind::Sos,
            });
        }

        let max_h = u32::from(header.sampling.max_h);
        let max_v = u32::from(header.sampling.max_v);
        let mcu_cols = info.dimensions.0.div_ceil(8 * max_h);
        let mcu_rows = info.dimensions.1.div_ceil(8 * max_v);
        let mut components = Vec::with_capacity(header.component_ids.len());
        for (output_index, &id) in header.component_ids.iter().enumerate() {
            let (h, v) =
                header
                    .sampling
                    .component(output_index)
                    .ok_or(JpegError::MissingMarker {
                        marker: MarkerKind::Sof,
                    })?;
            let quant_id =
                *header
                    .quant_table_ids
                    .get(output_index)
                    .ok_or(JpegError::MissingMarker {
                        marker: MarkerKind::Sof,
                    })? as usize;
            let quant = *header
                .quant_tables
                .entries
                .get(quant_id)
                .and_then(|q| q.as_ref())
                .ok_or(JpegError::MissingQuantTable {
                    component: id,
                    table_id: quant_id as u8,
                })?;
            components.push(PreparedProgressiveComponentPlan {
                h,
                v,
                output_index,
                quant: ctx.resolve_quant_table(quant),
                block_cols: mcu_cols * u32::from(h),
                block_rows: mcu_rows * u32::from(v),
                sample_width: info
                    .dimensions
                    .0
                    .saturating_mul(u32::from(h))
                    .div_ceil(max_h),
                sample_height: info
                    .dimensions
                    .1
                    .saturating_mul(u32::from(v))
                    .div_ceil(max_v),
            });
        }

        let mut scans = Vec::with_capacity(header.progressive_scans.len());
        for parsed in &header.progressive_scans {
            let mut scan_components = Vec::with_capacity(parsed.scan.components.len());
            for component in &parsed.scan.components {
                let component_index = find_component_index(&header.component_ids, component.id)
                    .ok_or(JpegError::UnknownScanComponent {
                        offset: parsed.entropy_offset,
                        component: component.id,
                    })?;
                let quant_id = *header.quant_table_ids.get(component_index).ok_or(
                    JpegError::MissingMarker {
                        marker: MarkerKind::Sof,
                    },
                )?;
                let _ = parsed
                    .quant_tables
                    .entries
                    .get(quant_id as usize)
                    .and_then(|q| q.as_ref())
                    .ok_or(JpegError::MissingQuantTable {
                        component: component.id,
                        table_id: quant_id,
                    })?;
                let dc_table = if parsed.scan.ss == 0 {
                    Some(resolve_progressive_huffman(
                        ctx,
                        &parsed.huffman_tables.dc,
                        component.id,
                        0,
                        component.dc_table,
                    )?)
                } else {
                    None
                };
                let ac_table = if parsed.scan.ss > 0 {
                    Some(resolve_progressive_huffman(
                        ctx,
                        &parsed.huffman_tables.ac,
                        component.id,
                        1,
                        component.ac_table,
                    )?)
                } else {
                    None
                };
                scan_components.push(PreparedProgressiveScanComponent {
                    component_index,
                    dc_table,
                    ac_table,
                });
            }
            scans.push(PreparedProgressiveScan {
                components: scan_components,
                ss: parsed.scan.ss,
                se: parsed.scan.se,
                ah: parsed.scan.ah,
                al: parsed.scan.al,
                entropy_offset: parsed.entropy_offset,
                restart_interval: parsed.restart_interval,
            });
        }

        let scratch_bytes =
            compute_progressive_scratch_bytes(&components, info.dimensions.0 as usize)?;
        Ok(PreparedProgressivePlan {
            components,
            scans,
            sampling: info.sampling,
            color_space: info.color_space,
            dimensions: info.dimensions,
            mcu_cols,
            mcu_rows,
            scratch_bytes,
        })
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "JPEG scan component counts are validated against the eight-bit SOS field"
    )]
    pub(super) fn build_progressive_host_output_plan(
        header: &ParsedHeader,
        info: &Info,
        ctx: &mut DecoderContext,
    ) -> Result<PreparedDecodePlan, JpegError> {
        let empty_raw = RawHuffmanTable {
            bits: [0; 16],
            values: HuffmanValues::default(),
        };
        let empty_huffman = ctx.resolve_huffman_table(&empty_raw)?;
        let mut components = Vec::with_capacity(header.component_ids.len());
        for (output_index, &id) in header.component_ids.iter().enumerate() {
            let (h, v) =
                header
                    .sampling
                    .component(output_index)
                    .ok_or(JpegError::MissingMarker {
                        marker: MarkerKind::Sof,
                    })?;
            let quant_id =
                *header
                    .quant_table_ids
                    .get(output_index)
                    .ok_or(JpegError::MissingMarker {
                        marker: MarkerKind::Sof,
                    })? as usize;
            let quant = *header
                .quant_tables
                .entries
                .get(quant_id)
                .and_then(|q| q.as_ref())
                .ok_or(JpegError::MissingQuantTable {
                    component: id,
                    table_id: quant_id as u8,
                })?;
            components.push(PreparedComponentPlan {
                h,
                v,
                output_index,
                quant: ctx.resolve_quant_table(quant),
                dc_table: Arc::clone(&empty_huffman),
                ac_table: Arc::clone(&empty_huffman),
            });
        }
        Ok(PreparedDecodePlan {
            components,
            sampling: info.sampling,
            color_space: info.color_space,
            restart_interval: header.restart_interval,
            dimensions: info.dimensions,
            scan_offset: header.sos_offset.ok_or(JpegError::MissingMarker {
                marker: MarkerKind::Sos,
            })?,
            scratch_bytes: compute_decode_scratch_bytes(
                info.dimensions,
                info.sampling,
                DEFAULT_MAX_DECODE_BYTES,
            )?,
        })
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "JPEG scan component counts are validated against the eight-bit SOS field"
)]
fn build_decode_plan(
    header: &ParsedHeader,
    info: &Info,
    dc_tables: &[Option<Arc<HuffmanTable>>; 4],
    ac_tables: &[Option<Arc<HuffmanTable>>; 4],
    ctx: &mut DecoderContext,
) -> Result<PreparedDecodePlan, JpegError> {
    let scan = header.scan.as_ref().ok_or(JpegError::MissingMarker {
        marker: MarkerKind::Sos,
    })?;
    let scan_offset = header.sos_offset.ok_or(JpegError::MissingMarker {
        marker: MarkerKind::Sos,
    })?;

    let mut components = Vec::with_capacity(scan.components.len());
    for scan_comp in scan.components.iter().copied() {
        let output_index = find_component_index(&header.component_ids, scan_comp.id).ok_or(
            JpegError::UnknownScanComponent {
                offset: scan_offset,
                component: scan_comp.id,
            },
        )?;
        let (h, v) = header
            .sampling
            .component(output_index)
            .ok_or(JpegError::MissingMarker {
                marker: MarkerKind::Sof,
            })?;
        let quant_id =
            *header
                .quant_table_ids
                .get(output_index)
                .ok_or(JpegError::MissingMarker {
                    marker: MarkerKind::Sof,
                })? as usize;
        let quant = *header
            .quant_tables
            .entries
            .get(quant_id)
            .and_then(|q| q.as_ref())
            .ok_or(JpegError::MissingQuantTable {
                component: scan_comp.id,
                table_id: quant_id as u8,
            })?;
        let dc_table = dc_tables[scan_comp.dc_table as usize].as_ref().ok_or(
            JpegError::MissingHuffmanTable {
                component: scan_comp.id,
                class: 0,
                id: scan_comp.dc_table,
            },
        )?;
        let ac_table = ac_tables[scan_comp.ac_table as usize].as_ref().ok_or(
            JpegError::MissingHuffmanTable {
                component: scan_comp.id,
                class: 1,
                id: scan_comp.ac_table,
            },
        )?;
        components.push(PreparedComponentPlan {
            h,
            v,
            output_index,
            quant: ctx.resolve_quant_table(quant),
            dc_table: Arc::clone(dc_table),
            ac_table: Arc::clone(ac_table),
        });
    }

    let mut scratch_bytes =
        compute_decode_scratch_bytes(info.dimensions, info.sampling, DEFAULT_MAX_DECODE_BYTES)?;
    if info.sof_kind == SofKind::Extended12 {
        // The sequential 12-bit paths render through full-frame u16 component
        // planes, which dwarf the stripe-based estimate above.
        scratch_bytes = scratch_bytes.max(compute_extended12_planes_scratch_bytes(
            &components,
            info.dimensions,
            info.sampling,
            DEFAULT_MAX_DECODE_BYTES,
        )?);
    }

    Ok(PreparedDecodePlan {
        components,
        sampling: info.sampling,
        color_space: info.color_space,
        restart_interval: header.restart_interval,
        dimensions: info.dimensions,
        scan_offset,
        scratch_bytes,
    })
}

fn validate_sampling_factors(header: &ParsedHeader, info: &Info) -> Result<(), JpegError> {
    validate_leading_component_sampling(header, info)?;
    for (h, v) in header.sampling.iter() {
        if h == 0 || v == 0 || h > 4 || v > 4 {
            return Err(JpegError::NotImplemented { sof: info.sof_kind });
        }
        if !header.sampling.max_h.is_multiple_of(h) || !header.sampling.max_v.is_multiple_of(v) {
            return Err(JpegError::NotImplemented { sof: info.sof_kind });
        }
    }
    Ok(())
}

fn validate_leading_component_sampling(
    header: &ParsedHeader,
    info: &Info,
) -> Result<(), JpegError> {
    if !matches!(info.color_space, ColorSpace::YCbCr) {
        return Ok(());
    }
    if let Some((h, v)) = header.sampling.component(0) {
        if h != header.sampling.max_h || v != header.sampling.max_v {
            return Err(JpegError::NotImplemented { sof: info.sof_kind });
        }
    }
    Ok(())
}

fn resolve_progressive_huffman(
    ctx: &mut DecoderContext,
    tables: &[Option<RawHuffmanTable>; 4],
    component: u8,
    class: u8,
    id: u8,
) -> Result<Arc<HuffmanTable>, JpegError> {
    let raw = tables
        .get(id as usize)
        .and_then(|table| table.as_ref())
        .ok_or(JpegError::MissingHuffmanTable {
            component,
            class,
            id,
        })?;
    ctx.resolve_huffman_table(raw)
}

fn compute_progressive_scratch_bytes(
    components: &[PreparedProgressiveComponentPlan],
    output_width: usize,
) -> Result<usize, JpegError> {
    let cap = DEFAULT_MAX_DECODE_BYTES;
    let mut total = 0usize;
    for component in components {
        let blocks = checked_usize_product(
            &[component.block_cols as usize, component.block_rows as usize],
            cap,
        )?;
        let coeffs = checked_usize_product(&[blocks, 64, core::mem::size_of::<i32>()], cap)?;
        total = total
            .checked_add(coeffs)
            .ok_or(JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap,
            })?;

        let plane = checked_usize_product(
            &[
                component.block_cols as usize,
                component.block_rows as usize,
                64,
            ],
            cap,
        )?;
        total = total
            .checked_add(plane)
            .ok_or(JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap,
            })?;
        if total > cap {
            return Err(JpegError::MemoryCapExceeded {
                requested: total,
                cap,
            });
        }
    }
    total =
        total
            .checked_add(output_width.saturating_mul(3))
            .ok_or(JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap,
            })?;
    if total > cap {
        return Err(JpegError::MemoryCapExceeded {
            requested: total,
            cap,
        });
    }
    Ok(total)
}

pub(super) fn find_component_index(component_ids: &[u8], id: u8) -> Option<usize> {
    component_ids
        .iter()
        .position(|&component_id| component_id == id)
}
