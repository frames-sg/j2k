# Changelog

All notable changes to this project will be documented in this file. The
format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Initial repository structure, workspace, and CI.
- Public API skeleton: `Decoder::inspect`, `Info`, `JpegError`, `Warning`.
- Marker-level header parser for SOF0/1/2/3, DQT, DHT, DRI, SOS, APP0, APP14.
