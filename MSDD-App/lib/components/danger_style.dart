import 'package:flutter/material.dart';

/// One place that decides how each danger level looks, so the map icons,
/// the zones list, the popup and the notifications always agree:
///
///   High (and Critical)  -> red     "Alert"
///   Medium               -> orange  "Warning"
///   anything else (Low)  -> yellow  "Notice"
class DangerStyle {
  DangerStyle._();

  static bool _isHigh(String level) => level == 'High' || level == 'Critical';

  static Color color(String level) {
    if (_isHigh(level)) return Colors.red;
    if (level == 'Medium') return Colors.orange;
    return Colors.yellow;
  }

  static IconData icon(String level) {
    if (_isHigh(level)) return Icons.error_rounded;
    if (level == 'Medium') return Icons.warning_amber_rounded;
    return Icons.info_outline_rounded;
  }

  /// "Alert", "Warning" or "Notice" — just the word, no level name.
  static String label(String level) {
    if (_isHigh(level)) return 'Alert';
    if (level == 'Medium') return 'Warning';
    return 'Notice';
  }
}
