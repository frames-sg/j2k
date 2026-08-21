// SPDX-License-Identifier: MIT OR Apache-2.0

//! Packet progression order and descriptor ordering.

use super::J2kPacketizationPacketDescriptor;

/// JPEG 2000 packet progression order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum J2kPacketizationProgressionOrder {
    /// Layer-resolution-component-position progression.
    Lrcp,
    /// Resolution-layer-component-position progression.
    Rlcp,
    /// Resolution-position-component-layer progression.
    Rpcl,
    /// Position-component-resolution-layer progression.
    Pcrl,
    /// Component-position-resolution-layer progression.
    Cprl,
}

impl J2kPacketizationProgressionOrder {
    /// Return the JPEG 2000 COD progression-order byte for this order.
    pub const fn codestream_order_code(self) -> u8 {
        match self {
            Self::Lrcp => 0x00,
            Self::Rlcp => 0x01,
            Self::Rpcl => 0x02,
            Self::Pcrl => 0x03,
            Self::Cprl => 0x04,
        }
    }
}

/// Sort explicit packet descriptors according to a JPEG 2000 progression order.
pub fn sort_packet_descriptors_for_progression(
    descriptors: &mut [J2kPacketizationPacketDescriptor],
    progression_order: J2kPacketizationProgressionOrder,
) {
    match progression_order {
        J2kPacketizationProgressionOrder::Lrcp => descriptors.sort_by_key(|descriptor| {
            (
                descriptor.layer,
                descriptor.resolution,
                descriptor.component,
                descriptor.precinct,
            )
        }),
        J2kPacketizationProgressionOrder::Rlcp => descriptors.sort_by_key(|descriptor| {
            (
                descriptor.resolution,
                descriptor.layer,
                descriptor.component,
                descriptor.precinct,
            )
        }),
        J2kPacketizationProgressionOrder::Rpcl => descriptors.sort_by_key(|descriptor| {
            (
                descriptor.resolution,
                descriptor.precinct,
                descriptor.component,
                descriptor.layer,
            )
        }),
        J2kPacketizationProgressionOrder::Pcrl => descriptors.sort_by_key(|descriptor| {
            (
                descriptor.precinct,
                descriptor.component,
                descriptor.resolution,
                descriptor.layer,
            )
        }),
        J2kPacketizationProgressionOrder::Cprl => descriptors.sort_by_key(|descriptor| {
            (
                descriptor.component,
                descriptor.precinct,
                descriptor.resolution,
                descriptor.layer,
            )
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        sort_packet_descriptors_for_progression, J2kPacketizationPacketDescriptor,
        J2kPacketizationProgressionOrder,
    };

    fn descriptors() -> [J2kPacketizationPacketDescriptor; 3] {
        [
            J2kPacketizationPacketDescriptor {
                packet_index: 0,
                state_index: 0,
                layer: 1,
                resolution: 0,
                component: 2,
                precinct: 1,
            },
            J2kPacketizationPacketDescriptor {
                packet_index: 1,
                state_index: 1,
                layer: 0,
                resolution: 1,
                component: 1,
                precinct: 0,
            },
            J2kPacketizationPacketDescriptor {
                packet_index: 2,
                state_index: 2,
                layer: 0,
                resolution: 0,
                component: 0,
                precinct: 2,
            },
        ]
    }

    #[test]
    fn progression_order_codes_match_codestream_values() {
        assert_eq!(
            J2kPacketizationProgressionOrder::Lrcp.codestream_order_code(),
            0
        );
        assert_eq!(
            J2kPacketizationProgressionOrder::Rlcp.codestream_order_code(),
            1
        );
        assert_eq!(
            J2kPacketizationProgressionOrder::Rpcl.codestream_order_code(),
            2
        );
        assert_eq!(
            J2kPacketizationProgressionOrder::Pcrl.codestream_order_code(),
            3
        );
        assert_eq!(
            J2kPacketizationProgressionOrder::Cprl.codestream_order_code(),
            4
        );
    }

    #[test]
    fn packet_descriptor_sort_uses_requested_progression_order() {
        let mut lrcp = descriptors();
        sort_packet_descriptors_for_progression(&mut lrcp, J2kPacketizationProgressionOrder::Lrcp);
        assert_eq!(lrcp.map(|descriptor| descriptor.packet_index), [2, 1, 0]);

        let mut pcrl = descriptors();
        sort_packet_descriptors_for_progression(&mut pcrl, J2kPacketizationProgressionOrder::Pcrl);
        assert_eq!(pcrl.map(|descriptor| descriptor.packet_index), [1, 0, 2]);
    }
}
