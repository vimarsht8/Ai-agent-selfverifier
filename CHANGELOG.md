echo '# Changelog

## [2.0.0] - 2024-01-XX

### Added
- Async/await support for better performance
- Persistent disk cache with TTL
- Chain-of-thought verification
- Calculator tool integration
- Metrics collection and monitoring
- Configuration management with Pydantic
- Support for OpenAI and Anthropic LLMs
- Streaming response support
- Comprehensive test suite

### Changed
- Refactored architecture for better modularity
- Improved error handling with retries
- Enhanced mock LLM for better testing

### Fixed
- Cache eviction bugs
- Import issues with List type hints

## [1.0.0] - 2024-01-XX

### Added
- Initial release with basic verification loop
- Support for OpenAI and Anthropic
- In-memory caching
- Command-line interface
' > CHANGELOG.md

# Add and push
git add CHANGELOG.md
git commit -m "Add CHANGELOG.md to track project history"
git push origin main