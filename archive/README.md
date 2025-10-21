# Archive - Previous Versions

This directory contains archived versions of BlackWall for reference and backwards compatibility.

## Version 3.4.0

Located in `v3.4.0/` directory.

This was the original monolithic implementation of BlackWall:
- **blackwall.py**: Original 866-line implementation
- **fix_dataset.py**: Dataset preprocessing utility
- **updatelog.md**: Version history and changes

### Key Features (v3.4.0):
- Basic ML model training
- Simulated network monitoring
- Simple CLI interface
- Dataset preprocessing

### Why It Was Archived:

Version 3.4.0 has been superseded by v4.0.0 which includes:
- Complete modular refactor
- Real packet capture
- Advanced threat detection
- REST API and web dashboard
- Database persistence
- Production-ready features

### Running v3.4.0:

If you need to run the old version:

```bash
cd archive/v3.4.0
python blackwall.py --help
```

**Note**: The old version requires the original requirements and may have dependencies that differ from v4.0.0.

---

For the latest version, use the main `blackwall.py` in the project root.
